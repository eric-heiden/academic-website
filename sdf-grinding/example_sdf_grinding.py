# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Grinding
#
# Removes workpiece volume by subtracting a moving grinding wheel from its
# texture SDF. Hydroelastic collision is evaluated without a dynamics solver.
#
# Command: python example_sdf_grinding.py
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

WORKPIECE_RADII = (0.45, 0.25, 0.12)
WORKPIECE_RESOLUTION = 256
GRINDER_RADIUS = 0.13
GRINDER_HALF_WIDTH = 0.04
GRIND_DEPTH = 0.035
HYDROELASTIC_STIFFNESS = 1.0e8

_SLOT_LINEAR = wp.uint32(0xFFFFFFFE)


@wp.func
def _sdf_cylinder_z(point: wp.vec3, radius: float, half_height: float) -> float:
    """Evaluate a capped cylinder aligned with its local Z-axis."""
    radial = wp.sqrt(point[0] * point[0] + point[1] * point[1]) - radius
    axial = wp.abs(point[2]) - half_height
    outside = wp.length(wp.vec2(wp.max(radial, 0.0), wp.max(axial, 0.0)))
    return outside + wp.min(wp.max(radial, axial), 0.0)


@wp.kernel(enable_backward=False)
def _subtract_cylinder_from_coarse_sdf(
    sdf_values: wp.array3d[wp.float32],
    sdf_lower: wp.vec3,
    voxel_size: wp.vec3,
    subgrid_size: int,
    grinder_xform: wp.transform,
    grinder_radius: float,
    grinder_half_width: float,
):
    z, y, x = wp.tid()
    grid_point = wp.vec3(float(x), float(y), float(z)) * float(subgrid_size)
    point = sdf_lower + wp.cw_mul(grid_point, voxel_size)
    grinder_point = wp.transform_point(wp.transform_inverse(grinder_xform), point)
    grinder_distance = _sdf_cylinder_z(grinder_point, grinder_radius, grinder_half_width)
    sdf_values[z, y, x] = wp.max(sdf_values[z, y, x], -grinder_distance)


@wp.kernel(enable_backward=False)
def _subtract_cylinder_from_subgrid_sdf(
    sdf_values: wp.array3d[wp.float32],
    subgrid_slots: wp.array3d[wp.uint32],
    sdf_lower: wp.vec3,
    voxel_size: wp.vec3,
    subgrid_size: int,
    grinder_xform: wp.transform,
    grinder_radius: float,
    grinder_half_width: float,
):
    tid = wp.tid()
    samples_per_dim = subgrid_size + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim
    block_id = tid // samples_per_subgrid
    sample_id = tid - block_id * samples_per_subgrid

    block_z = block_id // (subgrid_slots.shape[0] * subgrid_slots.shape[1])
    block_rem = block_id - block_z * subgrid_slots.shape[0] * subgrid_slots.shape[1]
    block_y = block_rem // subgrid_slots.shape[0]
    block_x = block_rem - block_y * subgrid_slots.shape[0]

    local_z = sample_id // (samples_per_dim * samples_per_dim)
    sample_rem = sample_id - local_z * samples_per_dim * samples_per_dim
    local_y = sample_rem // samples_per_dim
    local_x = sample_rem - local_y * samples_per_dim

    slot = subgrid_slots[block_x, block_y, block_z]
    if slot >= _SLOT_LINEAR:
        return
    address_x = int(slot & wp.uint32(0x3FF))
    address_y = int((slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    address_z = int((slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    texture_x = address_x * samples_per_dim + local_x
    texture_y = address_y * samples_per_dim + local_y
    texture_z = address_z * samples_per_dim + local_z

    fine_x = block_x * subgrid_size + local_x
    fine_y = block_y * subgrid_size + local_y
    fine_z = block_z * subgrid_size + local_z
    point = sdf_lower + wp.cw_mul(wp.vec3(float(fine_x), float(fine_y), float(fine_z)), voxel_size)
    grinder_point = wp.transform_point(wp.transform_inverse(grinder_xform), point)
    grinder_distance = _sdf_cylinder_z(grinder_point, grinder_radius, grinder_half_width)
    sdf_values[texture_z, texture_y, texture_x] = wp.max(sdf_values[texture_z, texture_y, texture_x], -grinder_distance)


@wp.kernel(enable_backward=False)
def _compute_normal_load(
    contact_count: wp.array[wp.int32],
    contact_distance: wp.array[wp.float32],
    contact_stiffness: wp.array[wp.float32],
    normal_load: wp.array[wp.float32],
):
    contact = wp.tid()
    if contact < contact_count[0]:
        normal_load[contact] = wp.max(-contact_distance[contact] * contact_stiffness[contact], 0.0)
    else:
        normal_load[contact] = 0.0


class Example:
    def __init__(self, viewer, _args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.frame = 0

        # A gently curved blank keeps fine SDF subgrids resident around the
        # entire machined surface, which makes this compact in-place demo
        # independent of Newton's linear-subgrid storage optimization.
        workpiece_mesh = newton.Mesh.create_ellipsoid(
            *WORKPIECE_RADII,
            num_latitudes=32,
            num_longitudes=64,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.workpiece_sdf = workpiece_mesh.build_sdf(
            max_resolution=WORKPIECE_RESOLUTION,
            narrow_band_range=(-0.08, 0.08),
            margin=0.04,
            texture_format="float32",
            paired_samples=False,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.005,
            kh=HYDROELASTIC_STIFFNESS,
            density=0.0,
            is_hydroelastic=True,
        )
        workpiece_cfg = shape_cfg.copy()
        workpiece_cfg.is_visible = False
        grinder_cfg = shape_cfg.copy()
        grinder_cfg.sdf_max_resolution = 64
        grinder_cfg.sdf_narrow_band_range = (-0.06, 0.06)
        grinder_cfg.sdf_padding = 0.01

        builder = newton.ModelBuilder()
        builder.sdf_texture_paired_samples = False
        builder.add_shape_mesh(
            body=-1,
            mesh=workpiece_mesh,
            cfg=workpiece_cfg,
            label="workpiece",
        )

        initial_pose = self._grinder_pose(0)
        self.grinder_body = builder.add_body(xform=initial_pose, label="grinder")
        builder.add_shape_cylinder(
            body=self.grinder_body,
            radius=GRINDER_RADIUS,
            half_height=GRINDER_HALF_WIDTH,
            cfg=grinder_cfg,
            color=(0.18, 0.20, 0.23),
            opacity=0.4,
            label="grinding_wheel",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=newton.geometry.HydroelasticSDF.Config(
                output_contact_surface=True,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.contact_surface = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
        self.contact_distance = wp.empty(
            self.contacts.rigid_contact_max,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.normal_load = wp.empty_like(self.contact_distance)
        self.total_normal_load = 0.0

        texture_data = self.workpiece_sdf.texture_data
        if texture_data is None:
            raise RuntimeError("The workpiece did not produce a texture SDF.")
        self.texture_data = texture_data
        self.coarse_values = wp.empty(
            (
                texture_data.coarse_texture.depth,
                texture_data.coarse_texture.height,
                texture_data.coarse_texture.width,
            ),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.subgrid_values = wp.empty(
            (
                texture_data.subgrid_texture.depth,
                texture_data.subgrid_texture.height,
                texture_data.subgrid_texture.width,
            ),
            dtype=wp.float32,
            device=self.model.device,
        )
        texture_data.coarse_texture.copy_to(self.coarse_values)
        texture_data.subgrid_texture.copy_to(self.subgrid_values)

        self.subgrid_work_items = texture_data.subgrid_start_slots.size * (int(texture_data.subgrid_size) + 1) ** 3

        self.body_q = self.state_0.body_q.numpy()
        self._set_grinder_pose(initial_pose)
        self._update_workpiece_surface()

        self.viewer.set_model(self.model)
        self.viewer.show_hydro_contact_surface = True
        self.viewer.set_camera(pos=wp.vec3(0.8, -0.85, 0.5), pitch=-18.0, yaw=132.0)

    @staticmethod
    def _grinder_pose(frame: int) -> wp.transform:
        x_start = -0.36
        x_end = 0.36
        x = min(x_start + frame * 0.004, x_end)
        radial_fraction = min((x / WORKPIECE_RADII[0]) ** 2, 1.0)
        surface_z = WORKPIECE_RADII[2] * math.sqrt(1.0 - radial_fraction)
        z = surface_z + GRINDER_RADIUS - GRIND_DEPTH
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        return wp.transform(wp.vec3(x, 0.0, z), rotation)

    def _set_grinder_pose(self, pose: wp.transform) -> None:
        self.body_q[self.grinder_body] = np.asarray(pose)
        self.state_0.body_q.assign(self.body_q)

    def _subtract_grinder(self, pose: wp.transform) -> None:
        wp.launch(
            _subtract_cylinder_from_coarse_sdf,
            dim=self.coarse_values.shape,
            inputs=[
                self.coarse_values,
                self.texture_data.sdf_box_lower,
                self.texture_data.voxel_size,
                int(self.texture_data.subgrid_size),
                pose,
                GRINDER_RADIUS,
                GRINDER_HALF_WIDTH,
            ],
            device=self.model.device,
        )
        wp.launch(
            _subtract_cylinder_from_subgrid_sdf,
            dim=self.subgrid_work_items,
            inputs=[
                self.subgrid_values,
                self.texture_data.subgrid_start_slots,
                self.texture_data.sdf_box_lower,
                self.texture_data.voxel_size,
                int(self.texture_data.subgrid_size),
                pose,
                GRINDER_RADIUS,
                GRINDER_HALF_WIDTH,
            ],
            device=self.model.device,
        )
        self.texture_data.coarse_texture.copy_from(self.coarse_values)
        self.texture_data.subgrid_texture.copy_from(self.subgrid_values)

    def _update_workpiece_surface(self) -> None:
        surface = self.workpiece_sdf.extract_isomesh(device=self.model.device)
        if surface is None:
            raise RuntimeError("Grinding removed the complete workpiece.")
        self.workpiece_points = wp.array(surface.vertices, dtype=wp.vec3, device=self.model.device)
        self.workpiece_indices = wp.array(surface.indices.reshape(-1), dtype=wp.int32, device=self.model.device)

    def step(self):
        self.frame += 1
        grinder_pose = self._grinder_pose(self.frame)
        self._set_grinder_pose(grinder_pose)

        # Run hydroelastic SDF-SDF collision before editing the workpiece.
        self.collision_pipeline.collide(self.state_0, self.contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state_0,
            self.contacts,
            out_distance=self.contact_distance,
        )
        wp.launch(
            _compute_normal_load,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_distance,
                self.contacts.rigid_contact_stiffness,
                self.normal_load,
            ],
            device=self.model.device,
        )
        self.total_normal_load = float(wp.utils.array_sum(self.normal_load))
        self._subtract_grinder(grinder_pose)
        self._update_workpiece_surface()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/workpiece_sdf",
            self.workpiece_points,
            self.workpiece_indices,
            color=(0.55, 0.62, 0.68),
            roughness=0.65,
            metallic=0.25,
            dynamic=True,
        )
        self.viewer.log_hydro_contact_surface(self.contact_surface)
        self.viewer.log_scalar("Hydroelastic normal load [N]", self.total_normal_load)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
