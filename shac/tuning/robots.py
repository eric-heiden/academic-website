"""Robot definitions used by the implicit-integration SHAC study."""

from __future__ import annotations

import dataclasses
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"
MJW = Path("/home/horde/repos/mujoco_warp-pr1535-357a75d")


@dataclasses.dataclass(frozen=True)
class Robot:
    name: str
    kind: str  # "quadruped" or "humanoid"
    xml: str
    keyframe: int | None
    # Termination: torso height window (m) and minimum body-z . world-z.
    min_height: float
    max_height: float
    min_up: float
    # Nominal standing torso height used by the height reward (m).
    target_height: float
    # Action a in [-1, 1] maps to ctrl = action_offset + action_scale * a.
    position_control: bool
    nconmax: int
    njmax: int
    # Optional default model edit (name in MODEL_EDITS) applied when the robot is built.
    model_edit: str | None = None


ROBOTS = {
    "ant": Robot(
        name="ant", kind="quadruped", xml=str(ROOT / "models" / "ant.xml"), keyframe=None,
        min_height=0.26, max_height=1.0, min_up=0.1, target_height=0.55,
        position_control=False, nconmax=32, njmax=128,
    ),
    "go1": Robot(
        name="go1", kind="quadruped", xml=str(MENAGERIE / "unitree_go1" / "scene.xml"), keyframe=0,
        min_height=0.18, max_height=0.6, min_up=0.5, target_height=0.27,
        position_control=True, nconmax=32, njmax=192,
    ),
    # Identical to go1 except that the joint damping is moved into the position
    # servos as actuator kv, a velocity-dependent actuator force.
    "go1_kv": Robot(
        name="go1_kv", kind="quadruped", xml=str(ROOT / "models" / "go1_kv" / "scene.xml"), keyframe=0,
        min_height=0.18, max_height=0.6, min_up=0.5, target_height=0.27,
        position_control=True, nconmax=32, njmax=192,
    ),
    "humanoid": Robot(
        name="humanoid", kind="humanoid", xml=str(ROOT / "models" / "humanoid.xml"), keyframe=None,
        min_height=0.74, max_height=2.1, min_up=0.0, target_height=1.3,
        position_control=False, nconmax=48, njmax=320,
    ),
    # MuJoCo Humanoid with actuator gears scaled by 1.75 to match the actuator
    # strength of the SHAC reference Humanoid (0.35 x motor_strengths vs 0.2 x).
    "humanoid_ref": Robot(
        name="humanoid_ref", kind="humanoid", xml=str(ROOT / "models" / "humanoid_ref.xml"), keyframe=None,
        min_height=0.74, max_height=2.1, min_up=0.0, target_height=1.3,
        position_control=False, nconmax=48, njmax=320,
    ),
    # H1 with the torque motors replaced by PD position servos (kp, kv inside the
    # actuator, torque limits equal to the original motor ranges).
    "h1_pd": Robot(
        name="h1_pd", kind="humanoid", xml=str(ROOT / "models" / "h1_pd" / "scene.xml"), keyframe=0,
        min_height=0.6, max_height=1.5, min_up=0.5, target_height=0.98,
        position_control=True, nconmax=48, njmax=320,
    ),
    "h1": Robot(
        name="h1", kind="humanoid", xml=str(MENAGERIE / "unitree_h1" / "scene.xml"), keyframe=0,
        min_height=0.6, max_height=1.5, min_up=0.5, target_height=0.98,
        position_control=False, nconmax=48, njmax=320,
    ),
}


# Model-editing functions referenced by SimConfig.model_edit (applied to the MjModel in place).
MODEL_EDITS: dict = {}


def _to_position_servos(model, gains: dict, pose: dict, spawn_clearance: float = 0.0):
    """Replaces torque motors by PD position servos and sets keyframe 0 to ``pose``.

    ``gains`` maps a joint-name suffix to (kp, kd, effort limit). The servo force is
    kp (ctrl - q) - kd qd, clamped to the effort limit; ctrl is limited to the joint range.
    The keyframe base height is chosen so that the lowest geom touches the floor, plus
    ``spawn_clearance``.
    """
    import mujoco
    import numpy as np

    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        key = next(k for k in gains if name.endswith(k))
        kp, kd, eff = gains[key]
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_biasprm[i, :] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_gear[i, :] = 0.0
        model.actuator_gear[i, 0] = 1.0
        model.actuator_ctrllimited[i] = 1
        model.actuator_ctrlrange[i] = model.jnt_range[j]
        model.actuator_forcelimited[i] = 1
        model.actuator_forcerange[i] = (-eff, eff)
    q = model.qpos0.copy()
    q[2] = 2.0
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for k, val in pose.items():
            if name.endswith(k):
                q[model.jnt_qposadr[j]] = val
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    # Lowest point of any collision geom (spheres/capsules/boxes approximated by their rbound).
    low = min(d.geom_xpos[g, 2] - (model.geom_size[g, 0] if model.geom_type[g] in (2, 3) else model.geom_rbound[g])
              for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
              if model.geom_bodyid[g] > 0)
    q[2] += -low + spawn_clearance
    if model.nkey == 0:
        raise ValueError("model needs a keyframe")
    model.key_qpos[0] = q
    model.key_qvel[0] = 0.0
    model.key_ctrl[0] = [q[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)]


def _floor_only(model):
    """Robot geoms collide with world geoms (the floor) only, as with Isaac Lab's disabled self-collision."""
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_bodyid[g] == 0:
            model.geom_contype[g], model.geom_conaffinity[g] = 1, 2
        else:
            model.geom_contype[g], model.geom_conaffinity[g] = 2, 1


def _joint_props(model, damping=None, armature=None, frictionloss=None):
    """Sets hinge-joint dof properties (the free joint is left unchanged)."""
    for j in range(1, model.njnt):
        dof = model.jnt_dofadr[j]
        if damping is not None:
            model.dof_damping[dof] = damping
        if armature is not None:
            model.dof_armature[dof] = armature
        if frictionloss is not None:
            model.dof_frictionloss[dof] = frictionloss


def apply_edits(model, names: str):
    """Applies '+'-separated MODEL_EDITS in order."""
    for n in names.split("+"):
        MODEL_EDITS[n](model)


MODEL_EDITS["floor_only"] = _floor_only

# Isaac Lab H1_CFG: implicit PD gains (stiffness, damping, effort limit) and default pose.
H1_IL_GAINS = {"hip_yaw": (150.0, 5.0, 300.0), "hip_roll": (150.0, 5.0, 300.0), "hip_pitch": (200.0, 5.0, 300.0),
               "knee": (200.0, 5.0, 300.0), "torso": (200.0, 5.0, 300.0), "ankle": (20.0, 4.0, 100.0),
               "shoulder_pitch": (40.0, 10.0, 300.0), "shoulder_roll": (40.0, 10.0, 300.0),
               "shoulder_yaw": (40.0, 10.0, 300.0), "elbow": (40.0, 10.0, 300.0)}
H1_IL_POSE = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -0.28, "knee": 0.79, "ankle": -0.52, "torso": 0.0,
              "shoulder_pitch": 0.28, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": 0.52}


def _h1_il(m):
    _joint_props(m, damping=0.0)  # the servo kd provides the damping (Isaac Lab implicit PD)
    _to_position_servos(m, H1_IL_GAINS, H1_IL_POSE)
    _floor_only(m)


MODEL_EDITS["h1_il"] = _h1_il

# Isaac Lab Go1 (flat) equivalent: DC-motor-like PD kp 25 / kd 0.5 (Go2 DCMotorCfg gains; the Go1 actuator
# net was fitted to a Kp 20 / Kd 0.5 controller), Newton-preset armature 0.02, no joint damping or friction
# loss, Menagerie force limits (23.7 N m hip/thigh, 35.55 N m calf), feet condim 3, floor-only collisions.
GO1_IL_GAINS = {"hip_joint": (25.0, 0.5, 23.7), "thigh_joint": (25.0, 0.5, 23.7), "calf_joint": (25.0, 0.5, 35.55)}
GO1_IL_POSE = {"FL_hip_joint": 0.1, "RL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
               "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
               "calf_joint": -1.5}


def _go1_il(m):
    import mujoco
    _joint_props(m, damping=0.0, armature=0.02, frictionloss=0.0)
    _to_position_servos(m, GO1_IL_GAINS, GO1_IL_POSE)
    for name in ("FR", "FL", "RR", "RL"):
        m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)] = 3
    _floor_only(m)


MODEL_EDITS["go1_il"] = _go1_il

ROBOTS["go1_il"] = Robot(
    name="go1_il", kind="quadruped", xml=str(MENAGERIE / "unitree_go1" / "scene.xml"), keyframe=0,
    min_height=0.12, max_height=0.6, min_up=0.0, target_height=0.28,
    position_control=True, nconmax=16, njmax=64, model_edit="go1_il",
)

ROBOTS["h1_il"] = Robot(
    name="h1_il", kind="humanoid", xml=str(MENAGERIE / "unitree_h1" / "scene.xml"), keyframe=0,
    min_height=0.6, max_height=1.5, min_up=0.5, target_height=0.98,
    position_control=True, nconmax=48, njmax=320, model_edit="h1_il",
)


def _g1_feet3(m):
    """Keeps 3 of the 7 foot capsules per foot (outer-left, centre, outer-right)."""
    import mujoco
    for side in ("left", "right"):
        for k in (1, 3, 5, 7):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{k}_collision")
            m.geom_contype[g] = m.geom_conaffinity[g] = 0


MODEL_EDITS["g1_feet3"] = _g1_feet3


def _g1_footpair(m):
    """Lets the left and right foot capsules collide with each other (capsule-capsule pairs, supported by the
    adjoint); all other robot geoms still collide with the floor only. Apply after floor_only and g1_feet3."""
    import mujoco
    for side, bit in (("left", "contype"), ("right", "conaffinity")):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                arr = m.geom_contype if bit == "contype" else m.geom_conaffinity
                arr[g] |= 4


MODEL_EDITS["g1_footpair"] = _g1_footpair

# Unitree G1 (29 dof) from mjlab via the MJWarp PR benchmark (capsule collision model, servo gains from motor
# armature and a 10 Hz natural frequency, actuator force limits), floor-only collisions, 3 capsules per foot.
ROBOTS["g1"] = Robot(
    name="g1", kind="humanoid", xml=str(ROOT / "models" / "g1_mjlab" / "scene_flat.xml"), keyframe=0,
    min_height=0.35, max_height=1.5, min_up=-1.0, target_height=0.78,
    position_control=True, nconmax=24, njmax=128, model_edit="floor_only+g1_feet3",
)


GO1_KP100_GAINS = {"hip_joint": (100.0, 2.0, 23.7), "thigh_joint": (100.0, 2.0, 23.7), "calf_joint": (100.0, 2.0, 35.55)}


def _go1_il_kp100(m):
    """go1_il with the Menagerie servo stiffness (kp 100) and its joint damping moved into kv (2)."""
    import mujoco
    _joint_props(m, damping=0.0, armature=0.02, frictionloss=0.0)
    _to_position_servos(m, GO1_KP100_GAINS, GO1_IL_POSE)
    for name in ("FR", "FL", "RR", "RL"):
        m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)] = 3
    _floor_only(m)


MODEL_EDITS["go1_il_kp100"] = _go1_il_kp100


# Isaac Lab G1_CFG gains (legs 150/200 kd 5; ankles 20/2; arms 40/10) and default pose on the mjlab G1.
G1_IL_GAINS = {"hip_yaw_joint": (150.0, 5.0, 88.0), "hip_roll_joint": (150.0, 5.0, 139.0),
               "hip_pitch_joint": (200.0, 5.0, 88.0), "knee_joint": (200.0, 5.0, 139.0),
               "ankle_pitch_joint": (20.0, 2.0, 50.0), "ankle_roll_joint": (20.0, 2.0, 50.0),
               "waist_yaw_joint": (200.0, 5.0, 88.0), "waist_roll_joint": (200.0, 5.0, 50.0),
               "waist_pitch_joint": (200.0, 5.0, 50.0), "shoulder_pitch_joint": (40.0, 10.0, 25.0),
               "shoulder_roll_joint": (40.0, 10.0, 25.0), "shoulder_yaw_joint": (40.0, 10.0, 25.0),
               "elbow_joint": (40.0, 10.0, 25.0), "wrist_roll_joint": (40.0, 10.0, 25.0),
               "wrist_pitch_joint": (40.0, 10.0, 5.0), "wrist_yaw_joint": (40.0, 10.0, 5.0)}
G1_IL_POSE = {"hip_pitch_joint": -0.20, "knee_joint": 0.42, "ankle_pitch_joint": -0.23, "elbow_joint": 0.87,
              "shoulder_pitch_joint": 0.35, "left_shoulder_roll_joint": 0.16, "right_shoulder_roll_joint": -0.16}


def _g1_il(m):
    _to_position_servos(m, G1_IL_GAINS, G1_IL_POSE)


MODEL_EDITS["g1_il"] = _g1_il


def _set_pose(model, pose: dict, spawn_clearance: float = 0.0):
    """Sets keyframe 0 to ``pose`` (joint-name suffix -> angle) with the lowest collision geom on the floor."""
    import mujoco
    q = model.key_qpos[0].copy() if model.nkey else model.qpos0.copy()
    q[2] = 2.0
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for k, val in pose.items():
            if name.endswith(k):
                q[model.jnt_qposadr[j]] = val
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    low = min(d.geom_xpos[g, 2] - (model.geom_size[g, 0] if model.geom_type[g] in (2, 3) else model.geom_rbound[g])
              for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g])
              and model.geom_bodyid[g] > 0)
    q[2] += -low + spawn_clearance
    model.key_qpos[0] = q
    model.key_ctrl[0] = [q[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)]


def _go1_servo_kv(m):
    """Menagerie Go1 with joint damping moved into the servos (kv), no friction loss, feet condim 3."""
    import mujoco
    for i in range(m.nu):
        dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        m.actuator_biasprm[i, 2] = -m.dof_damping[dof]
    _joint_props(m, damping=0.0, frictionloss=0.0)
    for name in ("FR", "FL", "RR", "RL"):
        m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)] = 3


def _go1_kp25(m):
    """Menagerie Go1 servos set to kp 25 / kv 0.5 (joint damping and friction loss kept)."""
    for i in range(m.nu):
        m.actuator_gainprm[i, 0] = 25.0
        m.actuator_biasprm[i, 1] = -25.0
        m.actuator_biasprm[i, 2] = -0.5


MODEL_EDITS["go1_pose_il"] = lambda m: _set_pose(m, GO1_IL_POSE)
MODEL_EDITS["arm02"] = lambda m: _joint_props(m, armature=0.02)
MODEL_EDITS["go1_servo_kv"] = _go1_servo_kv
MODEL_EDITS["go1_kp25"] = _go1_kp25


def _go1_condim3(m):
    import mujoco
    for name in ("FR", "FL", "RR", "RL"):
        m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)] = 3


def _damping_to_kv(m):
    """Moves joint damping into the actuator velocity gain (kv), leaving friction loss unchanged."""
    for i in range(m.nu):
        dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        m.actuator_biasprm[i, 2] = -m.dof_damping[dof]
    _joint_props(m, damping=0.0)


MODEL_EDITS["nofrictionloss"] = lambda m: _joint_props(m, frictionloss=0.0)
MODEL_EDITS["go1_condim3"] = _go1_condim3
MODEL_EDITS["damping_to_kv"] = _damping_to_kv


# Passive joint damping (N m s/rad) on all hinge joints, as dflex's always-on joint-limit damping (limit_kd=10).
MODEL_EDITS["damp10"] = lambda m: _joint_props(m, damping=10.0)
MODEL_EDITS["damp2"] = lambda m: _joint_props(m, damping=2.0)
