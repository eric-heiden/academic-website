You are calibrating a Newton physics simulation so that it reproduces reference photographs of a physical system.

Task: A red capsule pusher pushes a blue box (yellow marker on its +x end) across a table and stops. The box carries a hidden internal load. Calibrate table friction, pusher friction, and the box's in-plane center-of-mass offset.
Workspace: /home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0
- reference/reference.json: parameter names, bounds, units, training episodes, reference times, and the calibrated camera for every photo.
- reference/<episode>_<camera>_t<time>.png: reference photos; reference/sheet_<episode>.png: all photos of one episode.
- params.json: starting parameters. They are generic library defaults, not an informed guess.

Goal: find parameters for which the simulation reproduces the reference photos. Final quality is scored after you finish on held-out episodes with different initial conditions against hidden ground truth, so identify the physics rather than overfitting pixels. You have 30 minutes; working efficiently matters, and you should stop once the simulation matches the photos as well as you can make it.
Deliverable: write the final parameters as a JSON object containing every parameter name to /home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/params.json, then give a brief report.
Rules: work only inside this workspace plus the read-only Newton source tree /home/horde/apps/newton-live-mcp (and its docs). Do not search for or read hidden ground truth, other trials, or files elsewhere. Do not use subagents.

Workflow: the simulation is already running in a live Newton application, connected through the `newton` MCP server (tools: newton_execute, newton_observe, newton_describe, newton_rebuild). Images returned by MCP tools appear directly in your context, and Python state persists in the application between calls. You may also write and run your own scripts with the same task API (tools/mcp_evaluation/visual in the source tree) when that is more efficient; each script run starts a fresh simulator process.
Start by calling newton_describe once to confirm the connection. If no newton tools are available to you, reply only with NEWTON_TOOLS_UNAVAILABLE and stop.
Calibration task 'push'. Global `task` controls the scene (same API as the restart workflow):
- task.params; task.set_params({...}) rebuilds with new parameters and resets to t=0 (bounds: task.PARAMS).
- task.set_episode(name); training episodes ['push_a', 'push_b'].
- task.reset(); task.step(frames); task.simulate_to(t_seconds); frame dt 0.01 s.
- task.render(['top', 'side']) -> {camera: RGB uint8 array} from the reference cameras; task.rollout() renders all reference times.
- task.camera('top').observe_arguments() gives eye/target/up/fov_y/width/height for newton_observe/newton_filmstrip.
Reference photos: /home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/<episode>_<camera>_t<time>.png at times [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]; sheet_<episode>.png shows each episode.
Example comparison of the current parameters with the reference, one image:
newton_filmstrip(reset=true, times=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5], views=[<task.camera('top').observe_arguments()>], references=[["/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t0.00.png", "/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t0.25.png", "/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t0.50.png", "/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t0.75.png", "/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t1.00.png", "/home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/reference/push_a_top_t1.50.png"]])
(call task.set_episode(...) first for another episode). Write final parameters to /home/horde/artifacts/newton-live-mcp-v3/followup/push-opus-mcp_workers-0/params.json.
Parallel workers: `workers` is a pool of 3 more copies of this application, each with its own `task` scene and persistent Python state. Use them to evaluate candidates concurrently:
workers.broadcast("def evaluate(p):\n    task.set_params(p)\n    ...\n    return loss")  # define helpers on every worker once
losses = workers.map("result = evaluate(args)", [params_1, params_2, ...])  # runs in parallel, results in order
workers.submit(code, args) returns a future. Worker results must be JSON data; failed jobs return {"error": ...}.
