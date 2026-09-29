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
