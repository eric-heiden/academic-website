# Newton executor development evidence

This is an executor correctness and usability change. No timed agent comparison was run by this worker. Sources were based on `cd9bfdd8` in `/home/horde/apps/newton-live-mcp`; `session-before.py`, `protocol-before.py`, and `source-before-sha256.json` preserve the original sources. No source commit or push was made.

## Before and after

| Interaction | Before | After |
| --- | --- | --- |
| Define `import math`, `values`, and `total()`; call `result = total()` in another cell | NameError; fresh globals discarded the definitions and the error invalidated the scene | Result `6.0`; imports and function globals persist |
| Evaluate `2 + 3` | Result `null` | Result `5` |
| Define `measurements`, then raise `ZeroDivisionError` | Variables unavailable and ordinary execution blocked | Partial variables retained; default execution remains blocked; `recovery="inspect"` returns them while invalid; explicit `acknowledge` restores caller-accepted validity, always paused |
| Define a dataclass, `@wp.func`, and `@wp.kernel`, then launch in another cell | Warp cannot recover cell source (`OSError: could not get source code`) | Ordinary module identity and bounded source cache support definitions and later CPU launches |
| Display a one-million-element complex NumPy array | Automatic representation or conversion was unsafe/unhelpful | Compact shape/dtype summary; actual array retained in `_`; explicit oversized `result=` fails before dense conversion |

The exact baseline and replay interactions are in `interactions-before.json` and `interactions-after.json`. `kernel-and-display-after.json` records the dataclass/Warp and large-result examples on a real CPU session. Tests also validate functions seeing the current state after odd buffer swaps and stepping inside an execute cell.

## Semantics and lifecycle

- One registered Python module/global dictionary per session; imports, variables, functions, and classes persist across cells, physical reset, and checkpoint restore.
- Reserved live names `session`, `model`, `solver`, `state`, `state_next`, `control`, `contacts`, `viewer`, `wp`, and `np` refresh before each cell and after managed state changes. Normal user aliases and captured defaults remain ordinary references.
- The last expression returns automatically. Explicit `result=` retains precedence and strict JSON behavior, and is cleared before the next cell. `_` retains the last non-None actual value, including oversized/opaque values.
- `reset_namespace=True` clears Python definitions and source history before executing a compiled cell; syntax errors do not clear it. Replace/rebuild always clears the workspace. Close removes source history and the owned Python module registration.
- A single Warp module belongs to the workspace. Clear/close unloads compiled executables and drops its registered kernels/functions/structs. No new per-cell modules are retained.
- Source cache is bounded to 64 cells of at most 65536 characters. Successful responses and describe include generation, cell count, bounded variable names, and up to eight error stack frames with source lines.
- Compile errors cannot have run code and preserve physical validity. Runtime failures never assume rollback or infer safety from exception type: partial work remains, playback pauses, validity is false, and requires_rebuild is true.
- `recovery` is a closed JSON-schema enum: `none`, `inspect`, `acknowledge`. Inspect allows trusted diagnosis/repair while invalid without automatic recovery. A successful acknowledgement accepts the caller's assessment of coherence and always leaves playback paused. It does not restore arrays, notify solvers, clear caches, or prove correctness.
- Automatic display never invokes user repr or custom dictionary-key string conversion. Unsupported/oversized last expressions produce a bounded summary; callers can select attributes/slices explicitly. Conversion uses built-in JSON containers/scalar keys and NumPy values, with limits of 16384 components, 64 nesting levels, 1 MiB array conversion, and 65536 serialized characters. Stdout/stderr capture remains bounded to 16384 characters.
- Serialization failures after completed code preserve validity and saved values. The response explicitly says execution completed and advises against retrying the mutation.
- The existing full and code profiles, structured operations, opt-in permission, transport, and CLI remain compatible. No dependency or cwd change was introduced.

## Red/green validation

`tests-red.log` records the original executor failing new persistence, last-expression, lifecycle, recovery, source, and module-definition checks. `kernel-red.log` separately records actual Warp source lookup failing before the implementation. `display-red.log` records three review regressions before their fix: arbitrary repr executes and mutates/prints, large array display misses the bounded summary contract, and custom dictionary keys invoke user string conversion. These are behavior failures, not missing-module/import checks.

Final command:

```sh
uv run --no-sync --with mcp==1.26.0 -m unittest newton.tests.test_mcp_executor newton.tests.test_mcp -v
```

Result: **50 tests pass** (19 executor tests plus 31 existing runtime tests). See `tests-green.log`. Actual official MCP SDK stdio sessions validate schemas, imports/functions across cells, refreshed state, Warp definition/launch across cells, failure diagnostics, inspect/acknowledge recovery, and namespace clearing. SDK is a validation-only `--with` package, not a repository dependency. Existing runtime coverage includes CPU/CUDA stepping and notification regressions plus both MCP tool profiles.

```sh
uvx pre-commit run --files newton/_src/mcp/session.py newton/_src/mcp/protocol.py newton/tests/test_mcp_executor.py docs/guide/live_mcp.rst
uv run docs/generate_api.py
```

Targeted pre-commit passed (`pre-commit.log`); API generation completed without new generated symbol files because no public symbols were added. The public SimulationSession documentation and live MCP guide describe the lifecycle, result limits, and explicit recovery contract.

## Deliberate limits

Execution is synchronous full trusted Python, with no sandbox, IPython magics, top-level await, or safe forced interruption. Acknowledgement is an explicit caller assertion, not a coherence test. Ordinary aliases/default arguments, escaped references, separately named Warp modules, and application CUDA graphs remain application-owned. Clearing does not globally erase Warp module metadata or its disk cache. Source inspection of old functions can fail after source eviction, although those Python functions still execute. Persistent user objects consume memory until deleted or the namespace is cleared; no automatic rollback or notebook object eviction is claimed. Display summarizes opaque objects instead of calling their custom repr, so inspect selected fields explicitly when needed.
