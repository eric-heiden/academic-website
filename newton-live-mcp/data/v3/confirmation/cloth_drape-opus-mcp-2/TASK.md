You are calibrating a Newton physics simulation so that it reproduces reference photographs of a physical system.

Task: A 1 m square cloth is dropped flat onto a box (or other obstacle) and drapes under gravity. Calibrate the cloth material: stretch stiffness, bending stiffness, areal density, and friction.
Workspace: /home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2
- reference/reference.json: parameter names, bounds, units, training episodes, reference times, and the calibrated camera for every photo.
- reference/<episode>_<camera>_t<time>.png: reference photos; reference/sheet_<episode>.png: all photos of one episode.
- params.json: starting parameters. They are generic library defaults, not an informed guess.

Goal: find parameters for which the simulation reproduces the reference photos. Final quality is scored after you finish on held-out episodes with different initial conditions against hidden ground truth, so identify the physics rather than overfitting pixels. You have 30 minutes; working efficiently matters, and you should stop once the simulation matches the photos as well as you can make it.
Deliverable: write the final parameters as a JSON object containing every parameter name to /home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/params.json, then give a brief report.
Rules: work only inside this workspace plus the read-only Newton source tree /home/horde/apps/newton-live-mcp (and its docs). Do not search for or read hidden ground truth, other trials, or files elsewhere. Do not use subagents.

Workflow: the simulation is already running in a live Newton application, connected through the `newton` MCP server (tools: newton_execute, newton_observe, newton_filmstrip, newton_describe, newton_rebuild). Images returned by MCP tools appear directly in your context, and Python state persists in the application between calls. You may also write and run your own scripts with the same task API (tools/mcp_evaluation/visual in the source tree) when that is more efficient; each script run starts a fresh simulator process.
Start by calling newton_describe once to confirm the connection. If no newton tools are available to you, reply only with NEWTON_TOOLS_UNAVAILABLE and stop.
Calibration task 'cloth_drape'. Global `task` controls the scene (same API as the restart workflow):
- task.params; task.set_params({...}) rebuilds with new parameters and resets to t=0 (bounds: task.PARAMS).
- task.set_episode(name); training episodes ['center', 'corner'].
- task.reset(); task.step(frames); task.simulate_to(t_seconds); frame dt 0.0166667 s.
- task.render(['front', 'top']) -> {camera: RGB uint8 array} from the reference cameras; task.rollout() renders all reference times.
- task.camera('front').observe_arguments() gives eye/target/up/fov_y/width/height for newton_observe/newton_filmstrip.
Reference photos: /home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/reference/<episode>_<camera>_t<time>.png at times [0.5, 1.0, 2.0]; sheet_<episode>.png shows each episode.
Example comparison of the current parameters with the reference, one image:
newton_filmstrip(reset=true, times=[0.5, 1.0, 2.0], views=[<task.camera('front').observe_arguments()>], references=[["/home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/reference/center_front_t0.50.png", "/home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/reference/center_front_t1.00.png", "/home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/reference/center_front_t2.00.png"]])
(call task.set_episode(...) first for another episode). Write final parameters to /home/horde/artifacts/newton-live-mcp-v3/confirmation/cloth_drape-opus-mcp-2/params.json.
