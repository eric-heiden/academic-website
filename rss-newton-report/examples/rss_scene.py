"""A small manipulation scene for the RSS/Newton engineering study.

Run with RSS's live server; all positions passed as `at` use the bench frame.
No optional physics or rendering engine is required.
"""

import robosimstudio as rss

scene = rss.Scene(cell="bench_075", name="rss-newton-study")
scene.add(rss.robot("panda"))
scene.add(rss.box(name="cube", half=(0.035, 0.035, 0.035), at=(0.0, 0.17, 0.12), color=(0.15, 0.65, 0.9)))
scene.add(rss.sphere(name="ball", radius=0.035, at=(0.03, -0.2, 0.12), color=(0.95, 0.35, 0.15)))
scene.add(rss.sheet(name="towel", size=(0.24, 0.24), cells=(12, 12), at=(-0.12, -0.12, 0.05)))
scene.add(rss.CameraSpec(name="overview", eye=(1.7, 1.1, 1.65), look_at=(0.5, 0.0, 0.85)))

if __name__ == "__main__":
    rss.view(scene.build(control_mode="joint_position"), panels="all", share=False)
