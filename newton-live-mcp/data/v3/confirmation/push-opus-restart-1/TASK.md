You are calibrating a Newton physics simulation so that it reproduces reference photographs of a physical system.

Task: A red capsule pusher pushes a blue box (yellow marker on its +x end) across a table and stops. The box carries a hidden internal load. Calibrate table friction, pusher friction, and the box's in-plane center-of-mass offset.
Workspace: /home/horde/artifacts/newton-live-mcp-v3/confirmation/push-opus-restart-1
- reference/reference.json: parameter names, bounds, units, training episodes, reference times, and the calibrated camera for every photo.
- reference/<episode>_<camera>_t<time>.png: reference photos; reference/sheet_<episode>.png: all photos of one episode.
- params.json: starting parameters. They are generic library defaults, not an informed guess.

Goal: find parameters for which the simulation reproduces the reference photos. Final quality is scored after you finish on held-out episodes with different initial conditions against hidden ground truth, so identify the physics rather than overfitting pixels. You have 30 minutes; working efficiently matters, and you should stop once the simulation matches the photos as well as you can make it.
Deliverable: write the final parameters as a JSON object containing every parameter name to /home/horde/artifacts/newton-live-mcp-v3/confirmation/push-opus-restart-1/params.json, then give a brief report.
Rules: work only inside this workspace plus the read-only Newton source tree /home/horde/apps/newton-live-mcp (and its docs). Do not search for or read hidden ground truth, other trials, or files elsewhere. Do not use subagents.

Workflow: this is a script-based setup. Run
  uv run --no-sync --project /home/horde/apps/newton-live-mcp python sim.py
to simulate the training episodes with params.json. It writes out/compare_<episode>.png (rows per camera: simulated, reference, mismatch in magenta) and prints pixel statistics as JSON. Edit params.json (or pass --params other.json) and rerun; each run starts a fresh simulator process. You may also write your own scripts using the same task API that sim.py uses (tools/mcp_evaluation/visual in the source tree). Look at images with your image-viewing tool.
