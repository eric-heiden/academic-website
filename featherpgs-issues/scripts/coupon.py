"""Minimal parallel-gripper coupon: two pads squeeze one object and shake it.

Two variants of the same scene:

  "static"  - the pads are fixed to the world and the object is driven by an
              oscillating body force. Object/pad contacts take the free-rigid
              path, so this isolates contact and friction from articulation.

  "driven"  - the pads are the fingers of a three-joint articulation: a palm
              that slides along one axis to do the shaking, and two fingers
              that slide toward each other to do the squeezing, all position
              driven. Object/pad contacts take the articulated path, which is
              what a real gripper uses.

Both variants accept a primitive box or the same box as a triangle mesh.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

OBJ_HALF = 0.02
PAD_HALF_X = 0.005
PAD_HALF_Y = 0.05
PAD_HALF_Z = 0.015
DENSITY = 1000.0
Z0 = 0.10


def box_mesh(h):
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], dtype=np.float32)
    i = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                  0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
                  0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5], dtype=np.int32)
    return newton.Mesh(v, i)


def add_object(builder, geom, cfg):
    """The grasped object. add_body already gives it a free joint."""
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, Z0), wp.quat_identity()), label="object")
    if geom == "mesh":
        builder.add_shape_mesh(body, mesh=box_mesh(OBJ_HALF), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=OBJ_HALF, hy=OBJ_HALF, hz=OBJ_HALF, cfg=cfg)
    return body


def pad_contact_count(model, contacts, obj, n):
    """Contacts between the grasped object and anything else."""
    if n == 0:
        return 0
    sb = model.shape_body.numpy()
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    b0 = np.where(s0 >= 0, sb[np.clip(s0, 0, None)], -1)
    b1 = np.where(s1 >= 0, sb[np.clip(s1, 0, None)], -1)
    return int(((b0 == obj) | (b1 == obj)).sum())


@wp.kernel
def shake_force(body_f: wp.array[wp.spatial_vector], body: int, f: wp.vec3):
    body_f[body] = wp.spatial_vector(f, wp.vec3(0.0))


def build_static(geom="box", overlap=5.0e-4, mu=1.0, gap=5.0e-3):
    """Pads fixed to the world; the object is squeezed by a fixed overlap."""
    pad_x = OBJ_HALF + PAD_HALF_X - 0.5 * overlap
    cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=mu, ke=2.5e5,
                                          kd=1.0e3, gap=gap)
    b = newton.ModelBuilder()
    for x in (-pad_x, pad_x):
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(x, 0.0, Z0), wp.quat_identity()),
                        hx=PAD_HALF_X, hy=PAD_HALF_Y, hz=PAD_HALF_Z, cfg=cfg)
    obj = add_object(b, geom, cfg)
    return b, obj, None


def build_driven(geom="box", mu=1.0, gap=5.0e-3, squeeze=2.0e-3,
                 finger_ke=4.0e3, finger_kd=40.0, palm_ke=2.0e5, palm_kd=2.0e3,
                 lift=0.03):
    """A two-axis carrier holding two fingers, all position driven.

    The object starts resting on a fixed support, exactly as it would on a
    table, so the grasp has to close, lift it clear, and then hold it.
    """
    open_x = OBJ_HALF + PAD_HALF_X + 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=mu, ke=2.5e5,
                                          kd=1.0e3, gap=gap)
    b = newton.ModelBuilder()

    # fixed support the object rests on before the lift
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, Z0 - OBJ_HALF - 0.01),
                                           wp.quat_identity()),
                    hx=0.03, hy=0.05, hz=0.01, cfg=cfg)

    # carrier: one slide for the shake, one for the lift
    # a massless intermediate link is fine for FeatherPGS but SolverMuJoCo
    # rejects moving bodies without mass, so give it a token inertia
    carrier = b.add_link(label="carrier", mass=0.05,
                         inertia=wp.mat33(1.0e-4, 0.0, 0.0,
                                         0.0, 1.0e-4, 0.0,
                                         0.0, 0.0, 1.0e-4))
    joints = [b.add_joint_prismatic(
        -1, carrier, axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, Z0), wp.quat_identity()),
        target_ke=palm_ke, target_kd=palm_kd, armature=0.02, label="shake_x")]

    # The palm is a mount, not a collider: it carries no geometry, so it cannot
    # touch the object. Its mass matches the 100 x 100 x 20 mm plate it replaces
    # so the carrier dynamics are unchanged.
    palm = b.add_link(label="palm", mass=0.2,
                      inertia=wp.mat33(1.733e-4, 0.0, 0.0,
                                       0.0, 1.733e-4, 0.0,
                                       0.0, 0.0, 3.333e-4))
    joints.append(b.add_joint_prismatic(
        carrier, palm, axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=palm_ke, target_kd=palm_kd, armature=0.02, label="lift_z"))

    fingers = []
    for sign, name in ((-1.0, "finger_l"), (1.0, "finger_r")):
        f = b.add_link(label=name)
        b.add_shape_box(f, hx=PAD_HALF_X, hy=PAD_HALF_Y, hz=PAD_HALF_Z, cfg=cfg)
        joints.append(b.add_joint_prismatic(
            palm, f, axis=wp.vec3(-sign, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(sign * open_x, 0.0, 0.0),
                                      wp.quat_identity()),
            target_ke=finger_ke, target_kd=finger_kd, armature=0.001, label=name))
        fingers.append(f)
    b.add_articulation(joints, label="gripper")

    obj = add_object(b, geom, cfg)
    close_target = open_x - (OBJ_HALF + PAD_HALF_X) + squeeze
    return b, obj, {"palm": palm, "carrier": carrier, "fingers": fingers,
                    "close": close_target, "lift": lift}


def make(kind, geom="box", solver="fpgs", solver_kw=None, gravity=True, **kw):
    if kind == "static":
        b, obj, drive = build_static(geom, **kw)
    else:
        b, obj, drive = build_driven(geom, **kw)
    if not gravity:
        b.gravity = 0.0
    model = b.finalize()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=4096,
                                        broad_phase="nxn")
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    if solver == "fpgs":
        cfg = dict(pgs_mode="matrix_free", pgs_iterations=12, pgs_beta=0.2,
                   pgs_cfm=1.0e-6, dense_max_constraints=256,
                   mf_max_constraints=4096, double_buffer=False)
        cfg.update(solver_kw or {})
        solv = newton.solvers.SolverFeatherPGS(model, **cfg)
        control._use_coord_layout_targets = False
    else:
        cfg = dict(solver="newton", integrator="implicitfast", iterations=15,
                   ls_iterations=50, nconmax=4096, njmax=8192,
                   cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        cfg.update(solver_kw or {})
        solv = newton.solvers.SolverMuJoCo(model, **cfg)
    return dict(model=model, pipeline=pipeline, contacts=contacts, state=[s0, s1],
                control=control, solver=solv, obj=obj, drive=drive, cfg=cfg)


def run(env, frames=600, substeps=16, amplitude=0.03, frequency=2.0,
        close_frames=45, lift_frames=45, settle_frames=30, record_poses=False):
    """Close, lift clear of the support, settle, then shake."""
    model, pipeline, contacts = env["model"], env["pipeline"], env["contacts"]
    state, control, solver = env["state"], env["control"], env["solver"]
    obj, drive = env["obj"], env["drive"]
    dt = 1.0 / 60.0 / substeps
    mass = float(model.body_mass.numpy()[obj])
    omega = 2.0 * math.pi * frequency
    targets = control.joint_target_q.numpy().copy() if drive else None
    start_shake = close_frames + lift_frames + settle_frames

    log = {k: [] for k in ("t", "obj_pos", "obj_quat", "carrier_x", "lift_z",
                           "finger_q", "finger_target", "n_contacts", "pad_contacts")}
    poses = []
    ref = None
    failure = ""
    try:
        for frame in range(frames):
            t_frame = frame / 60.0
            shaking = frame >= start_shake
            ts = (frame - start_shake) / 60.0
            if drive is not None:
                close = min(1.0, frame / max(1.0, close_frames))
                rise = min(1.0, max(0.0, frame - close_frames) / max(1.0, lift_frames))
                targets[0] = amplitude * math.sin(omega * ts) if shaking else 0.0
                targets[1] = drive["lift"] * rise
                targets[2] = targets[3] = drive["close"] * close
                control.joint_target_q.assign(targets.astype(np.float32))

            pipeline.collide(state[0], contacts)
            for k in range(substeps):
                state[0].clear_forces()
                if drive is None and shaking:
                    a = -mass * amplitude * omega * omega * math.sin(omega * (ts + k * dt))
                    wp.launch(shake_force, dim=1,
                              inputs=[state[0].body_f, obj, wp.vec3(0.0, a, 0.0)])
                solver.step(state[0], state[1], control, contacts, dt)
                state[0], state[1] = state[1], state[0]

            bq = state[0].body_q.numpy()
            p = bq[obj, :3].copy()
            # slip is measured against the gripper, which moves in the driven case
            origin = bq[drive["palm"], :3].copy() if drive else np.zeros(3)
            rel = p - origin
            if shaking and ref is None:
                ref = rel.copy()
            jq = state[0].joint_q.numpy()
            log["t"].append(t_frame)
            log["obj_pos"].append(rel.tolist())
            log["obj_quat"].append(bq[obj, 3:7].tolist())
            log["carrier_x"].append(float(origin[0]))
            log["lift_z"].append(float(origin[2]))
            log["finger_q"].append([float(jq[2]), float(jq[3])] if drive else [0.0, 0.0])
            log["finger_target"].append([float(targets[2]), float(targets[3])]
                                        if drive else [0.0, 0.0])
            n = int(contacts.rigid_contact_count.numpy()[0])
            log["n_contacts"].append(n)
            log["pad_contacts"].append(pad_contact_count(model, contacts, obj, n))
            if record_poses:
                poses.append(bq.copy())
            if not np.all(np.isfinite(p)):
                failure = f"non-finite pose at frame {frame}"
                break
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"

    out = {k: np.asarray(v) for k, v in log.items()}
    out["failure"] = failure
    out["start_shake"] = start_shake
    if record_poses:
        out["poses"] = np.stack(poses) if poses else np.zeros((0, 1, 7))
    if ref is None or len(out["obj_pos"]) == 0:
        out["held"] = False
        return out
    pos = out["obj_pos"]
    shake = np.arange(len(pos)) >= start_shake
    d = pos[shake] - ref
    lifted = out["lift_z"][shake][0] > 0.5 * (drive["lift"] if drive else 1.0)
    out["held"] = bool(np.all(np.isfinite(pos[-1])) and abs(d[-1][2]) < 0.02
                       and (lifted or drive is None))
    out["slip_mm"] = float(np.abs(d).max() * 1000.0)
    out["axial_slip_mm"] = float(np.abs(d[:, 2]).max() * 1000.0)
    out["lateral_mm"] = float(np.abs(d[:, [0, 1]]).max() * 1000.0)
    out["final_slip_mm"] = float(np.linalg.norm(d[-1]) * 1000.0)
    out["contacts_median"] = float(np.median(out["pad_contacts"][shake]))
    out["zero_contact_frames"] = int((out["pad_contacts"][shake] == 0).sum())
    out["shake_frames"] = int(shake.sum())
    if drive is not None:
        out["finger_overshoot_mm"] = float(
            np.mean(out["finger_target"][shake] - out["finger_q"][shake]) * 1000.0)
    return out
