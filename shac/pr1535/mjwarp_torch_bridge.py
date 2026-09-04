"""Reusable PyTorch autograd bridge for MJWarp PR #1535.

The PR records its analytic adjoint only for the out-of-place
``mujoco_warp.step(model, data_in, data_out)`` form. This bridge replays one
such step during PyTorch backward, seeds the output-state adjoints explicitly,
and returns gradients for ``qpos``, ``qvel``, and ``ctrl``.

Only ``qpos`` and ``qvel`` are treated as carried differentiable state.
Other state, notably the solver warm-start, remains detached in the template
input data. The bridge is reusable but intentionally single-threaded.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


def _sync(device: torch.device) -> None:
    wp.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _MJWarpStep(torch.autograd.Function):
    """One custom-autograd node backed by PR #1535's analytic step VJP."""

    @staticmethod
    def forward(
        ctx,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
        bridge: MJWarpTorchBridge,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bridge._check_inputs(qpos, qvel, ctrl)
        ctx.bridge = bridge
        ctx.save_for_backward(qpos, qvel, ctrl)
        return bridge._forward_raw(qpos, qvel, ctrl)

    @staticmethod
    def backward(
        ctx,
        grad_qpos_out: torch.Tensor | None,
        grad_qvel_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        qpos, qvel, ctrl = ctx.saved_tensors
        dqpos, dqvel, dctrl = ctx.bridge._backward_raw(
            qpos,
            qvel,
            ctrl,
            grad_qpos_out,
            grad_qvel_out,
        )
        return dqpos, dqvel, dctrl, None


class MJWarpTorchBridge:
    """Reusable, single-threaded one-step bridge from PyTorch to MJWarp."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        *,
        nworld: int,
        device: str,
        nconmax: int = 64,
        njmax: int = 256,
    ) -> None:
        self.nworld = nworld
        self.nq = mj_model.nq
        self.nv = mj_model.nv
        self.nu = mj_model.nu
        self.wp_device = wp.get_device(device)
        self.torch_device = torch.device(str(self.wp_device))

        with wp.ScopedDevice(self.wp_device):
            self.model = mjw.put_model(mj_model)
            kwargs = {
                "nworld": nworld,
                "nconmax": nconmax,
                "njmax": njmax,
            }
            self.data_in = mjw.put_data(mj_model, mj_data, **kwargs)
            self.data_out = mjw.put_data(mj_model, mj_data, **kwargs)

            for data in (self.data_in, self.data_out):
                data.qpos.requires_grad = True
                data.qvel.requires_grad = True
            self.data_in.ctrl.requires_grad = True

            # Build and validate the reusable scratch before any timed backward.
            self.backward_workspace = mjw.create_backward_context(
                self.model, self.data_in
            )

    def _check_inputs(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> None:
        expected = (
            (qpos, (self.nworld, self.nq), "qpos"),
            (qvel, (self.nworld, self.nv), "qvel"),
            (ctrl, (self.nworld, self.nu), "ctrl"),
        )
        for value, shape, name in expected:
            if value.shape != shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {shape}"
                )
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must be float32, got {value.dtype}")
            if value.device != self.torch_device:
                raise ValueError(
                    f"{name} is on {value.device}, expected {self.torch_device}"
                )

    @staticmethod
    def _torch_to_warp(value: torch.Tensor, *, requires_grad: bool = False) -> wp.array:
        return wp.from_torch(
            value.detach().contiguous(), dtype=wp.float32, requires_grad=requires_grad
        )

    @staticmethod
    def _zero_grad(array: wp.array) -> None:
        if array.grad is not None:
            array.grad.zero_()

    def _load_inputs(
        self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor
    ) -> None:
        wp.copy(self.data_in.qpos, self._torch_to_warp(qpos))
        wp.copy(self.data_in.qvel, self._torch_to_warp(qvel))
        wp.copy(self.data_in.ctrl, self._torch_to_warp(ctrl))

    def step(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable one-step map usable inside a PyTorch computation."""
        return _MJWarpStep.apply(qpos, qvel, ctrl, self)

    def _forward_raw(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the same step without recording a PyTorch autograd node."""
        self._check_inputs(qpos, qvel, ctrl)
        with wp.ScopedDevice(self.wp_device):
            self._load_inputs(qpos, qvel, ctrl)
            mjw.step(self.model, self.data_in, self.data_out)
            _sync(self.torch_device)
            qpos_out = wp.to_torch(self.data_out.qpos).detach().clone()
            qvel_out = wp.to_torch(self.data_out.qvel).detach().clone()
        return qpos_out, qvel_out

    def _backward_raw(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
        grad_qpos_out: torch.Tensor | None,
        grad_qvel_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with wp.ScopedDevice(self.wp_device):
            self._load_inputs(qpos, qvel, ctrl)
            for array in (
                self.data_in.qpos,
                self.data_in.qvel,
                self.data_in.ctrl,
                self.data_out.qpos,
                self.data_out.qvel,
            ):
                self._zero_grad(array)

            tape = wp.Tape()
            with tape:
                # The third argument is essential: PR #1535 deliberately does
                # not record its analytic backward for the in-place overload.
                mjw.step(self.model, self.data_in, self.data_out)

            if grad_qpos_out is None:
                grad_qpos_out = torch.zeros_like(qpos)
            if grad_qvel_out is None:
                grad_qvel_out = torch.zeros_like(qvel)
            seeds = {
                self.data_out.qpos: self._torch_to_warp(grad_qpos_out),
                self.data_out.qvel: self._torch_to_warp(grad_qvel_out),
            }
            with mjw.backward_context(self.backward_workspace):
                tape.backward(grads=seeds)
            _sync(self.torch_device)

            dqpos = wp.to_torch(self.data_in.qpos.grad).detach().clone()
            dqvel = wp.to_torch(self.data_in.qvel.grad).detach().clone()
            dctrl = wp.to_torch(self.data_in.ctrl.grad).detach().clone()
        return dqpos, dqvel, dctrl
