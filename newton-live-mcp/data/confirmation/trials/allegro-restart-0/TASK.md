You are an independent GPT-6 Astra evaluation agent. Solve this Newton simulation task in this workspace.

Task: Track smooth finger flexion and hold a final posture with the Menagerie Allegro hand.
Variant: 0. Time budget: 600 seconds, maximum 12 full candidate rollouts.
Read task.json for parameter bounds and fixed quality thresholds. Success requires all thresholds and 1500 finite steps. Submit final parameters by editing this workspace's config.py CONFIG dict. Do not modify the reference trajectory, dynamics, scoring, task files, imported assets, or shared implementation. Only config.py is editable. You may inspect the common Newton and harness source files, official docs, and local source assets for this task. Do not read other trial directories, scenario feasibility results, other agents' conversations, or tuned answers. No subagents.

Both conditions have the same physical simulator, targets, measurements, images, parameter ranges, and final fresh-process verification. Use observations if useful; avoid unnecessary expensive rendering. Keep all runs and failed attempts. Do not claim success without completing a measured rollout. Finish with a brief report of your config and measured quality. Quality is independently verified after your process exits.

Common source directory: <newton>/tools/mcp_evaluation
Simulation backend: Newton SolverMuJoCo CPU with native MuJoCo contacts. Images use the same Newton sensor renderer. A generated collision-pipeline contact query is a diagnostic, distinct from native solver contacts. Duration 3 seconds, dt .002 seconds.

Edit config.py, then run a fresh process per candidate:
uv run --no-sync --project <newton> python rollout.py --scenario allegro --variant 0 --config config.py --output metrics.json
Add --observe to save an image after a rollout. Read metrics.json and provenance.json. Each invocation must exit after its single rollout. You may batch independent candidates with one new process each. Do not keep a simulator process alive across candidates or use live MCP.
