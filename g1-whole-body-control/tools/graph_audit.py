"""Inspect the optimizer on 64-bit Linux using the CUDA driver API.

Run from Newton: uv run python graph_audit.py --motion walk.csv --output audit.json
"""

import ctypes as C
import json
import time
from collections import Counter
from pathlib import Path

import newton.viewer
import warp as wp
from newton.examples.robot.example_robot_g1_wbc import Example

a = Example.create_parser().parse_args()
if not a.output or C.sizeof(C.c_void_p) != 8:
    raise ValueError("Supply --output on a 64-bit Linux CUDA host")
e = Example(newton.viewer.ViewerNull(), a)
cuda = C.CDLL("libcuda.so.1")
cuda.cuGraphGetNodes.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_size_t),
]
cuda.cuGraphNodeGetType.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
cuda.cuGraphKernelNodeGetParams.argtypes = [C.c_void_p, C.c_void_p]
cuda.cuGraphMemcpyNodeGetParams.argtypes = [C.c_void_p, C.c_void_p]
cuda.cuFuncGetName.argtypes = [C.POINTER(C.c_char_p), C.c_void_p]


def check(result):
    if result:
        raise RuntimeError(f"CUDA driver status {result}")


def inspect_graph(graph):
    n = C.c_size_t()
    check(cuda.cuGraphGetNodes(graph.graph, None, C.byref(n)))
    nodes = (C.c_void_p * n.value)()
    check(cuda.cuGraphGetNodes(graph.graph, nodes, C.byref(n)))
    types = Counter()
    names = Counter()
    copies = Counter()
    for handle in nodes:
        kind = C.c_int()
        check(cuda.cuGraphNodeGetType(C.c_void_p(handle), C.byref(kind)))
        types[kind.value] += 1
        if kind.value == 1:
            params = C.create_string_buffer(256)
            check(cuda.cuGraphMemcpyNodeGetParams(C.c_void_p(handle), C.byref(params)))
            # CUDA_MEMCPY3D ABI offsets for src/dstMemoryType on this 64-bit host.
            src = C.c_int.from_buffer(params, 32).value
            dst = C.c_int.from_buffer(params, 120).value
            copies[f"{src}->{dst}"] += 1
        if kind.value == 0:
            params = C.create_string_buffer(256)
            check(cuda.cuGraphKernelNodeGetParams(C.c_void_p(handle), C.byref(params)))
            func = C.c_void_p.from_buffer(params)
            name = C.c_char_p()
            check(cuda.cuFuncGetName(C.byref(name), func))
            names[name.value.decode()] += 1
    assert types.get(3, 0) == 0, "Host callback in optimizer graph"
    assert all(kind == "2->2" for kind in copies), (
        "Non-device memcpy in optimizer graph"
    )
    return {
        "node_count": n.value,
        "node_types": dict(types),
        "memcpy_memory_types": dict(copies),
        "kernel_names": dict(names),
    }


results = {
    "regular": inspect_graph(e.mpc.graph),
    "config": vars(a),
    "device": e.model.device.name,
    "scope": "Top-level graph nodes; conditional-node bodies are not enumerated.",
}
if hasattr(e.mpc, "initial_graph"):
    results["initial"] = inspect_graph(e.mpc.initial_graph)
# Compare complete graph replay and eager submission on a fixed state.
for mode in ("graph", "eager"):
    times = []
    for repeat in range(5):
        e.mpc.plan.zero_()
        e.mpc.center.zero_()
        e.mpc.iteration.zero_()
        e.mpc.last.zero_()
        wp.record_event(e.start_event)
        start = time.perf_counter()
        if mode == "graph":
            wp.capture_launch(e.mpc.graph)
        else:
            e.mpc.optimize(
                e.solver.mjw_data.qpos, e.solver.mjw_data.qvel, e.solver.mjw_data.time
            )
        wp.record_event(e.end_event)
        cost = e.mpc.minimum.numpy()
        times.append(
            {
                "gpu_ms": wp.get_event_elapsed_time(e.start_event, e.end_event),
                "wall_ms": 1000 * (time.perf_counter() - start),
                "cost": float(cost[0]),
            }
        )
    results[mode] = times
Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
print(
    json.dumps({k: v for k, v in results.items() if k not in ("regular", "initial")}),
    flush=True,
)
