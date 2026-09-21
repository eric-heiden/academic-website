You are an independent GPT-6 Astra evaluation agent. Solve this Newton simulation task in this workspace.

Task: Track a smooth seven-joint motion under gravity with the Menagerie Panda arm.
Variant: 1. Time budget: 600 seconds, maximum 12 full candidate rollouts.
Read task.json for parameter bounds and fixed quality thresholds. Success requires all thresholds and 1500 finite steps. Submit final parameters by editing this workspace's config.py CONFIG dict. Do not modify the reference trajectory, dynamics, scoring, task files, imported assets, or shared implementation. Only config.py is editable. You may inspect the common Newton and harness source files, official docs, and local source assets for this task. Do not read other trial directories, scenario feasibility results, other agents' conversations, or tuned answers. No subagents.

All conditions have the same physical simulator, targets, measurements, images, parameter ranges, and final fresh-process verification. Use observations if useful; avoid unnecessary expensive rendering. Keep all runs and failed attempts. Do not claim success without completing a measured rollout. Finish with a brief report of your config and measured quality. Quality is independently verified after your process exits.

Common source directory: /home/horde/apps/newton-live-mcp/tools/mcp_evaluation
Simulation backend: Newton SolverMuJoCo CPU with native MuJoCo contacts. Images use the same Newton sensor renderer. A generated collision-pipeline contact query is a diagnostic, distinct from native solver contacts. Duration 3 seconds, dt .002 seconds.

The same Newton application is running inside a fresh IPython kernel. The upstream ipython-mcp server is configured. Call connect_to_kernel(connection_file='/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda-ipython-1/kernel.json') once, then execute_code(code=...) for actual Python access. Do not start a second kernel or simulator. The kernel contains session, scenario, model, solver, state, state_next, control, contacts, np and wp. User variables, imports and functions persist across calls; live application aliases refresh between cells after state swaps and rebuilds.
For one complete candidate, batching in one code call is allowed and efficient:
session.scenario.apply_config({...})
session.dispatch('reset')
session.dispatch('step', {'count': 1500})
session.scenario.metrics()
The final expression is displayed; unlike Newton MCP, assigning result alone does not display it. You may loop over candidates and return or print a compact list, within the same total 12-rollout budget. All shared structured helpers remain available through session.dispatch; direct Python application inspection is allowed.
Use session.dispatch('observe', camera_settings) to generate the same sensor observations, save the returned image_base64 as a PNG and inspect that file with your image tool. Avoid printing image_base64. For HUG, session.scenario.provenance exposes the source/frame setup. Do not call scenario.rollout directly, modify scoring, or mutate state/targets. Candidate parameter changes use apply_config so final settings are reproducible.
The upstream server waits 30 seconds for a code reply internally. Keep each execution within that duration; splitting a batch is allowed. A timeout does not prove execution stopped: do not automatically repeat mutations. MCP tool timeout is 300 seconds. Ordinary Python errors retain the IPython namespace and may have partially mutated the scene. Use session.dispatch('rebuild') if recovery is necessary; it retains the latest validated configuration by default. Do not read or print the kernel connection file; pass its path to connect_to_kernel.
After success, write the exact final configuration to config.py for independent fresh-process verification. Use actual IPython MCP tools for running simulation access; do not connect to the kernel using a separate shell client.
