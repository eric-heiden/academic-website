"""RSS scene, explicit Newton solver routing, cameras, and interactive studio.

Run in the RSS checkout: uv run /absolute/path/to/rss_cloth_cameras.py
Also usable with robosimstudio_agent.serve --scene ... --camera-view.
"""
import robosimstudio as rss
from robosimstudio_studio import StudioConfig

# GENERIC is an ArrangementSpec; backend names select solver factories.
physics = rss.replace(
    rss.GENERIC,
    control_hz=60, substeps=8,
    backends=(("articulated", rss.backend("mujoco")),
              ("*", rss.backend("vbd"))),
)
scene = rss.Scene(cell="bench_075", physics=physics, name="cloth-cameras")
scene.add(rss.robot("panda"))
scene.add(rss.sheet(name="towel", size=(0.32, 0.32), cells=(18, 18),
                    at=(-0.16, -0.16, 0.05)))
scene.add(rss.box(name="block", half=(0.035,) * 3, at=(0.02, 0.25, 0.06)))
# Camera poses use world coordinates; `at` above is relative to the bench.
scene.add(rss.CameraSpec(name="front", width=640, height=400,
                         eye=(1.3, -0.7, 1.2), look_at=(0.5, 0, 0.85)))
scene.add(rss.CameraSpec(name="overhead", width=640, height=400,
                         eye=(0.5, 0.01, 1.8), look_at=(0.5, 0, 0.8)))

if __name__ == "__main__":
    # build creates the Newton models, state/control, solver pipeline and warmup.
    sim = scene.build(control_mode="joint_position")
    print(sim.pipeline.program.summary())
    # Live camera panel owns its tiled renderer; the browser runs on Viser.
    rss.view(sim, panels="all", config=StudioConfig(live=True, port=8097), share=False)
