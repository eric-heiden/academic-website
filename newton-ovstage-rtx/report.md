# USD scene preservation for Newton RTX

Newton’s USD importer reconstructs a limited visual scene, which can discard authored appearance. Meanwhile, ovnewton imports physics from ovstage through a second implementation of many of Newton’s import rules. This report explains the two workflows and proposes separate ownership for scene preservation and physics interpretation: retain the composed scene for OVRTX, centralize physics policy in Newton, and keep ovnewton as an optional adapter. Isaac Lab asset comparisons and synthetic pose replay support the rendering approach; complete task compatibility and the shared importer remain unproven. Publication overhead is substantial for 64 moving robots.

<p class="prototype-note"><a href="https://github.com/eric-heiden/newton/tree/prototype/ovstage-viewer">Newton prototype branch</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/docs/prototypes/ovstage-viewer.md">API and reproduction</a>.<br><strong>Implemented:</strong> external-scene rendering and CPU/CUDA pose transport. <strong>Proposed:</strong> the complete viewer upgrade, dependency changes and shared physics importer.</p>

## 1. Motivation and component roles

USD (Universal Scene Description) combines referenced assets, overrides, materials, lights and physics into a composed scene. Newton converts that scene into a simulation model and portable visual shapes. Reconstructing an RTX scene from those shapes can lose material networks, subdivision settings or environment content. [Issue #4051](https://github.com/newton-physics/newton/issues/4051) requests visual and lighting parity with Omniverse Kit; extending Newton’s material parser addresses individual losses but also gives the physics engine more rendering semantics to maintain.

**The proposal addresses two separate problems:** preserve authored appearance for rendering, and eliminate duplicated physics interpretation. ovstage enables the first; adopting it does not automatically solve the second.

| Component | Responsibility |
|---|---|
| **ovpopulation — load** | Population component that uses OpenUSD to load composed rendering and/or physics data into ovstage. Python entry point: `ovstage.population.open_usd(...)`. |
| **ovstage — share** | Runtime scene store for prims (scene objects), hierarchy, attributes and tensor access. Increasing publication ordinals identify updates and coordinate when consumers may read them. It is neither a physics solver nor a Newton model builder. |
| **ovnewton — connect** | Imports ovstage physics into Newton and synchronizes simulation state with the stage. It currently contains its own physics interpretation; the proposal replaces that policy with a shared Newton implementation. |
| **OVRTX / ViewerRTX — render / display** | OVRTX renders an attached ovstage. Newton’s ViewerRTX provides the viewer interface and displays render outputs. Neither requires ovnewton merely to render a stage. |

The intended division follows OVRTX’s [external-stage architecture](https://nvidia-omniverse.github.io/ovrtx/core/ovstage_integration.html). OpenUSD still handles USD composition through [population](https://github.com/NVIDIA-Omniverse/ovstage/blob/71f917ad449d104fb062c6149221cda13417814b/docs/scene/population.rst); Newton should interpret the physics needed for simulation, while the rendering backend interprets supported visual schemas.

## 2. Current workflows and duplicated parsing

**Ordinary Newton application:** call `ModelBuilder.add_usd(...)`, finalize the model, then run a solver and viewer. Newton’s OpenUSD importer reads physics and extracts portable visuals. The existing generated-scene ViewerRTX path builds a new render scene from those visuals.

**Application using ovnewton:** populate an ovstage, import it with `ovnewton.add_ovstage(...)`, finalize a Newton model, and attach a state binding. The application runs the solver and publishes its results to the same stage that OVRTX renders. These are alternative import routes; an application need not run both physics importers.

<!-- CURRENT_FLOW -->

### How an application uses ovnewton today

```python
builder = newton.ModelBuilder()
result = ovnewton.add_ovstage(builder, stage, ordinal=ordinal)
model = builder.finalize(
    skip_validation_joints=result.has_orphan_joints,
)
binding = ovnewton.attach_ovstage(
    stage, model=model, ordinal=ordinal,
)

# After the solver advances state for a frame:
ordinal += 1
binding.update_to_ovstage(state, ordinal=ordinal)
stage.advance_write_floor(ordinal).wait()
renderer.step(
    render_products={product}, delta_time=dt, ordinal=ordinal,
)
```

<p class="caption">Current API excerpt from the inspected <a href="https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/examples/example_ovnewton_basic.py">ovnewton example</a> (repository access required). Assumes a populated, published stage, renderer, render product, solver state and frame duration <code>dt</code>. Register solver-specific attributes before import. <code>add_ovstage</code> currently requires an otherwise empty builder; <code>attach_ovstage(model=...)</code> binds an existing model, while omitting <code>model</code> imports and finalizes one for convenience.</p>

“Parsing ovstage” means querying runtime attributes and interpreting their physics meaning, rather than opening USD files again in Python. The inspected implementation separates this work into three layers:

| Layer | What ovnewton does today |
|---|---|
| **Read — `_stage.py`** | Queries prim paths and fixed-length or variable-length attribute columns at a published ordinal; fetches groups and copies values into CPU arrays. This is a source reader, not a second USD composition engine. |
| **Interpret — `_parse.py`** | Extracts units, gravity, bodies, materials, colliders, joints and hierarchy; derives relationships and some converted values. |
| **Build — `_build.py`** | Chooses Newton geometry, resolves mass/inertia and material defaults, creates bodies and joints, and applies initial state. These decisions overlap with Newton’s OpenUSD importer. |

For example, both paths decide how authored mass interacts with density-derived mass and missing inertia, how units and joint angles are converted, and how colliders and joints become Newton objects. A policy change can therefore require two fixes and two sets of coverage. The duplication is **physics semantics across two implementations**, even if each reads its own source efficiently. [Newton importer](https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/newton/_src/utils/import_usd.py), [ovnewton interpretation](https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/_parse.py), [model construction](https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/_build.py).

Per-frame synchronization is separate from import. ovnewton publishes body poses, velocities and supported joint state, and can read supported stage state and drive targets back into Newton. It does not rebuild the physics model each frame. The current binding uses body/joint labels as prim paths and must be recreated after incompatible topology changes. [Binding contract](https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/ovnewton.py).

## 3. Proposed scene and viewer ownership

Preserve the composed render scene. Newton imports physics and publishes motion through a body-to-prim mapping, which associates simulated bodies with scene objects. **Keep scene ownership and live USD prims outside `Model`; retain portable visuals for other viewers.**

A source prim path on each visual shape is useful provenance but insufficient as the scene contract: it can omit lights, environment and material-binding context. Storing live `UsdPrim` handles also ties their validity to the source stage’s lifetime. Simulation bodies and render prims need not correspond one-to-one, especially after body collapse or instancing. An application-owned stage plus explicit bindings preserves that distinction.

<!-- RUNTIME_FLOW -->

**ViewerRTX can remain independent of ovnewton.** Put shared pose transport in Newton’s optional ovstage integration; ovnewton delegates pose writes to it. OVRTX renders; ViewerRTX presents the result.

| Viewer mode | Scene ownership |
|---|---|
| **Existing USD / Isaac Lab scene** | Application owns stage, renderer and publication; ViewerRTX borrows them. |
| **Procedural Newton example** | Viewer creates its stage, geometry, camera and lights; uses the same transport and render path. |

The prototype implements borrowed-scene rendering and explicit CPU/CUDA pose bindings. Its ordinary physics import can still use Newton’s existing OpenUSD importer while population loads the render scene. **This proves the rendering boundary, not unified import or a single startup traversal.**

## 4. Shared physics import and future usage

**One physics interpretation, two source readers.** Newton owns units, mass/inertia, collision geometry and joint construction. The OpenUSD reader and ovnewton’s ovstage reader supply source facts to that implementation. Moving ovnewton’s parser into Newton while retaining the old parser would preserve the duplication.

<!-- IMPORT_FLOW -->

The reader contract must carry values, units, hierarchy, stable source identities and whether a value was authored, inherited/defaulted or blocked. A resolved number alone can lose information needed to choose a Newton default. Readers also need a common representation of instances and their transforms. ovstage does not need to acquire Newton-specific model-building policy.

| Application | Proposed user-facing workflow |
|---|---|
| **USD file → Newton** | Keep `builder.add_usd(...)` as the convenience entry point. It delegates physics construction to the shared importer; portable viewers remain available. An RTX application additionally preserves the source scene. |
| **Existing ovstage → Newton** | Keep the `ovnewton.add_ovstage(...)` / finalize / attach lifecycle above. Internally, the adapter reads ovstage facts and calls the same Newton importer instead of its own physics policy. |
| **Procedural Newton model → RTX** | ViewerRTX creates an ovstage and uses shared pose transport. No ovnewton import is needed. |

**Dependency direction:** ovnewton depends on Newton; Newton’s optional RTX integration depends on ovstage/OVRTX. Newton does not depend on ovnewton. ovnewton retains its useful state/control synchronization and convenience API, delegating common pose writes to Newton’s transport. Sharing pose transport alone does not unify the physics importers.

**Mapping contract:** keep a binding object beside the model with prim paths, stable body identities and rest transforms. Resolve final body indices after construction, and remap bindings when models are cloned, bodies are collapsed or arrays are reordered. The prototype requires final indices explicitly; automatic remapping is proposed work. The application owns stage lifetime and publication.

| Order / owner | Deliverable | Done when |
|---|---|---|
| **1 · Newton** | Complete the viewer upgrade in §5; keep the existing USD importer during transition. | Both viewer modes pass headless/window, camera, overlay, deformation and cleanup checks on the qualified runtime pair. |
| **2 · Newton + ovnewton** | Define source facts and remappable bindings; extract one physics importer in Newton and replace both existing policy paths with calls to it. ovnewton reuses pose transport. | The two readers produce equivalent models for mass/density/inertia, units, joints and defaults; mapping tests cover cloning, collapse and reordering. |
| **3 · ovstage + ovnewton** | Close value-provenance and instanced-collider gaps; align supported Newton and ovstage versions. | ANYmal, Franka and Cartpole import correctly with one jointly tested dependency set. |

<p class="caption">Steps 2–3 proceed together. Inspected ovnewton pins Newton <code>&gt;=1.5,&lt;1.6</code> and ovstage <code>0.2.0.378985</code>; it cannot simply be added to Newton 1.7 development. That alignment is separate from the viewer-only runtime pair below. These are findings at the linked source revision, not claims about every later release.</p>

## 5. OVRTX upgrade in Newton

**Migrate the viewer before unifying the importers.** Newton’s `rtx` extra allows `ovrtx>=0.3`; its generated-scene path still calls deprecated renderer scene APIs. A version bump alone leaves that migration unfinished. [Current code](https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/newton/_src/viewer/viewer_rtx.py), [OVRTX architecture](https://nvidia-omniverse.github.io/ovrtx/core/ovstage_integration.html).

| Change in Newton | Concrete replacement |
|---|---|
| **Qualify the runtime pair** | Start with tested `ovrtx==0.5.0.377615` + `ovstage==0.2.0.377349` in optional `newton[rtx]`; replace the open-ended renderer requirement. Core stays independent. |
| **Separate scene creation from display** | Remove `ViewerRTX`’s `ViewerUSD` inheritance. Compose an owned-scene builder or borrowed scene with one render-product consumer. Existing USD generation can bootstrap owned scenes once. |
| **Move scene operations to ovstage** | Replace renderer `open_usd`, bindings and attribute writes with population, cached stage queries and tensor writes. Port cameras, visibility, meshes and debug overlays too. |
| **Adopt the 0.5 interfaces** | Use DLPack tensor interchange, Boolean reset flags and full RenderVar output paths; remove `.tensor` wrappers and raw `DLTensor` construction. |

<div class="code-pair" markdown="1">

### Load once — current APIs

```python
ovrtx.register_schema_paths()
renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(sync_mode=True))
stage = ovstage.Stage("scene")

renderer.attach_ovstage(stage)
ovstage.population.open_usd(
    stage,
    scene_usd,
    ordinal=1,
    domains=ovstage.PopulationDomain.ALL,
)
stage.advance_write_floor(1).wait()
```

### Publish, then render — prototype API

```python
binding = OvstageBodyBinding(
    stage,
    model,
    ordinal=1,
    prim_paths=paths,
    body_indices=indices,
    body_local_transforms=offsets,
)
viewer = ViewerRTX(
    stage=stage,
    renderer=renderer,
    render_product=product,
)

# Finish rendering before the next write.
binding.write(state, ordinal=frame)
stage.advance_write_floor(frame).wait()
viewer.render(ordinal=frame)  # frame starts at 2
```

</div>

<p class="caption">Excerpts assume an authored camera/RenderProduct and final-model mappings. Register additional physics schemas before population. See <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/docs/prototypes/ovstage-viewer.md">imports and cleanup</a> and <a href="https://github.com/NVIDIA-Omniverse/ovrtx/blob/e3ebb35a6024d070fe21125f3806d6152ed3c753/CHANGELOG.md">0.5 API changes</a>. The prototype window is a fixed-camera preview; full interaction still needs porting.</p>

## 6. Visual comparison

Both sides use OVRTX 0.5 with matched camera, lighting and authored settings. These are Isaac Lab asset-derived scenes, including a generated heightfield; **complete reinforcement-learning (RL) tasks and parity with Omniverse Kit rendering remain untested**. Select a scene and compare the overlay.

<!-- COMPARISONS -->

Preserving the source addresses appearance discarded during import in [#4051](https://github.com/newton-physics/newton/issues/4051). Exact Kit matching still depends on backend features, exposure, color management and temporal settings.

## 7. Performance and qualification

| Measured result | Required follow-up |
|---|---|
| **41–46 ms/frame** for 64 moving ANYmals, before physics | Profile publication before treating this as suitable for large RL workloads. |
| **20–21 ms** in global publication alone | ovstage/OVRTX team: reduce commit cost and retest moving scenes. |
| **+164–812 MiB GPU allocation** for small source scenes | Set workload-specific memory budgets; preserved appearance has a measurable cost. |
| Earlier import: **52.5 → 7.6 s** without visual extraction | Promising duplicate-work reduction; separate OVRTX 0.3 experiment, not a clone-pipeline comparison. |

<p class="caption">13 research runs · RTX 4090 / Windows · 640 × 480 · ≥30 s warmup + 60 frames. Synthetic pose replay, not solver execution; mostly one process repetition. GPU figures are allocation snapshots. Prototype: 64 robots at 41.4 ms/frame, including 22.5 ms encoding/writes/publication, without importing PXR or ovnewton in the replay’s Python process. Population still uses OpenUSD internally.</p>

<details class="evidence"><summary>Inspect all 13 runs and measurement definitions</summary>

<!-- NEW_MEASUREMENTS -->

</details>

**Release the viewer upgrade after its functional checks. Qualify Isaac Lab separately:** repeated timing with physics, peak CPU/GPU memory, cloning, multiple cameras and matched Kit captures. These establish the workload limits.

<details class="evidence"><summary>Sources, prototype code and reproduction</summary>
<ul>
<li><a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/docs/prototypes/ovstage-viewer.md">Prototype design and reproduction</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/newton/tests/test_viewer_rtx_stage.py">Ownership and CPU/CUDA tests</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/pyproject.toml">Current Newton dependencies</a>.</li>
<li><a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/newton/_src/viewer/ovstage.py">Pose bridge</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/newton/_src/viewer/rtx_scene.py">Render-product consumer</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/scripts/prototypes/rtx_ovstage.py">Scene replay script</a>.</li>
<li><a href="https://github.com/NVIDIA-Omniverse/ovstage/blob/71f917ad449d104fb062c6149221cda13417814b/README.md">ovstage overview</a> · <a href="https://github.com/NVIDIA-Omniverse/ovstage/blob/71f917ad449d104fb062c6149221cda13417814b/docs/scene/population.rst">Population API</a> · <a href="https://github.com/NVIDIA-Omniverse/ovstage/blob/71f917ad449d104fb062c6149221cda13417814b/CHANGELOG.md">ovstage 0.2 changes</a>.</li>
<li><a href="https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/ovnewton.py">Inspected ovnewton entry points</a> · <a href="https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/pyproject.toml">ovnewton dependency pins</a> (repository access required).</li>
<li><a href="https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/_stage.py">ovnewton column reader</a> · <a href="https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/fe5dcfe9e017e2e462ede0ab7dea73120dd0b46d/ovnewton/_src/_runtime.py">State transport</a> (repository access required).</li>
<li><a href="https://github.com/newton-physics/newton/pull/4170">PR #4170</a> and <a href="https://github.com/newton-physics/newton/pull/4177">PR #4177</a> can still benefit portable viewers; native RTX appearance should come from the source scene.</li>
</ul>
</details>
