You are an independent GPT-6 Astra evaluation agent. Solve this Newton simulation task in this workspace.

Task: Reconstruct a recorded HUG MANO approach to a scanned softball, then tune dynamic replay stability and wrist tracking.
Variant: 0. Time budget: 600 seconds, maximum 12 full candidate rollouts.
Read task.json for parameter bounds and fixed quality thresholds. Success requires all thresholds and 1500 finite steps. Submit final parameters by editing this workspace's config.py CONFIG dict. Do not modify the reference trajectory, dynamics, scoring, task files, imported assets, or shared implementation. Only config.py is editable. You may inspect the common Newton and harness source files, official docs, and local source assets for this task. Do not read other trial directories, scenario feasibility results, other agents' conversations, or tuned answers. No subagents.

All conditions have the same physical simulator, targets, measurements, images, parameter ranges, and final fresh-process verification. Use observations if useful; avoid unnecessary expensive rendering. Keep all runs and failed attempts. Do not claim success without completing a measured rollout. Finish with a brief report of your config and measured quality. Quality is independently verified after your process exits.

Common source directory: /home/horde/apps/newton-live-mcp/tools/mcp_evaluation
Simulation backend: Newton SolverMuJoCo CPU with native MuJoCo contacts. Images use the same Newton sensor renderer. A generated collision-pipeline contact query is a diagnostic, distinct from native solver contacts. Duration 3 seconds, dt .002 seconds.

The Newton application is already running and the newton MCP server is configured with a compact code profile: describe, execute, observe and rebuild. You must use actual MCP tools for live queries and rollout control. Discovery is optional; use describe or session.dispatch('query', {...}) inside execute when useful. Use execute for the application's validated parameter interface:
result = session.scenario.apply_config({...})
For one complete candidate, batching in one execute call is allowed and efficient:
session.scenario.apply_config({...})
session.dispatch('reset')
session.dispatch('step', {'count': 1500})
result = session.scenario.metrics()
All structured operations remain available through session.dispatch inside execute. Multiple candidates may be batched within the same total 12-rollout budget.
The step operation advances the application's fixed targets and scoring. Use the observe MCP tool directly with the camera from task.json if useful; it returns an image content block. For HUG, execute result = session.scenario.provenance exposes the source/frame setup. Do not call scenario.rollout directly, modify scoring, or mutate state/targets. You may use dispatch query/edit to inspect or demonstrate model parameter handling, but candidate parameter changes should use apply_config so final settings are reproducible. Python imports, variables and functions persist across execute calls; the final expression is returned, or assign result explicitly. Execution errors preserve variables but pause and invalidate the simulation. Use execute with recovery="inspect" to diagnose, or recovery="acknowledge" only after verifying or repairing coherent state. Alternatively use rebuild in the same process; it retains the last validated configuration by default and clears the Python workspace. Optional rebuild arguments may contain a config dict. MCP startup is bounded at 30 seconds and each tool call at 300 seconds; a timed-out running mutation has an unknown outcome and must not be automatically retried. After success, write the exact final configuration to config.py for the independent fresh-process verification.
