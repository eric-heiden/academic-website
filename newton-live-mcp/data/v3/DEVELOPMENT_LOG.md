# v3 development log (not confirmation evidence)

## Pilot round 1 (2026-09-28/29), MCP condition forbade separate scripts for push, allowed afterwards

| Trial | Held-out | Time [s] | Input tok | Tool calls | Notes |
|---|---|---|---|---|---|
| push opus mcp | fail (2.8x pos) | 519 | 476k | 17 | fitted training frames to ~2 mm but converged to wrong COM x / pusher mu |
| push opus restart | pass | 336 | 402k | 14 | fit.py + two parallel Nelder-Mead background processes; `log=False` bypassed candidate log |
| push astra mcp | pass (0.25) | 1053 | 1.56M | 22 | LHS + least squares in live app |
| push astra restart | pass (0.62) | 1151 | 3.74M | 76 | 55 of 76 calls were tail/cat polling of background CMA-ES logs |
| cloth opus mcp | pass (0.57) | 349 | 469k | 17 | show() composites for visual friction check; 1 process |
| cloth opus restart | pass (0.59) | 483 | 605k | 19 | 13 simulator processes |
| cloth astra mcp | pass (0.49) | 1129 | 5.10M | 78 | Codex session had NO newton MCP tools (agent built raw TCP client); infra failure |
| cloth astra restart | timeout | 1200 | lost | 89 | would pass held-out; Codex usage lost on kill |

## Findings -> changes
- Agents read the task source and bypassed candidate logging (`log=False`) -> logging is harness-controlled only.
- Restart agents parallelize/background optimizers; live app is single-threaded -> MCP condition now allows scripts
  too (question = does adding MCP help a coding agent), and we measure MCP usage.
- Script workflows cost many polling calls (Astra push restart: 55 inspection calls) -> main expected MCP benefit is
  fewer round trips/tokens, not simulator startup (3-5 s per process here).
- Codex intermittently starts without MCP tools (reproduced 1/5 with a minimal prompt) -> MCP prompt requires a first
  newton_describe; token NEWTON_TOOLS_UNAVAILABLE; harness retries infra failures (retained), Claude init status checked.
- Astra xhigh routinely needs >20 min; Codex loses usage when killed -> budget 30 min, SIGINT before kill.
- Filmstrip reference times must be multiples of the frame dt -> push frame dt 0.01 s, references regenerated.
- Warp kernel-load chatter in stdout -> wp.config.quiet in shared task module.
- Workspaces must use absolute paths (one relaunch after workspaces were created inside the repo; runs killed, deleted).

## Restart cost context (warm kernel cache, this machine, 2 frames)
cloth_h1 (H1 + garment) 20.8 s, nut_bolt_hydro 10.3 s, mpm_granular 7.8 s, cloth_franka 8.8 s, nut_bolt_sdf 7.3 s,
cloth_hanging 4.9 s; our tasks 3-5 s. H1 garment: ~85 s per 2 s simulated (too slow for agent iteration).
