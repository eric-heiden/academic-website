# USD scene preservation for Newton RTX

Newton’s USD importer reconstructs a limited visual scene, which can discard authored appearance. This study compares that path with rendering the composed source scene through ovstage and OVRTX, using Isaac Lab assets and synthetic pose replay. Source preservation retains visual information without adding renderer-specific shader parsing to Newton. The proposed migration keeps ViewerRTX, centralizes physics interpretation in Newton and uses ovnewton as an optional adapter. Publication overhead remains substantial for 64 moving robots.

<p class="prototype-note"><a href="https://github.com/eric-heiden/newton/tree/prototype/ovstage-viewer">Newton prototype branch</a> · <a href="https://github.com/eric-heiden/newton/blob/prototype/ovstage-viewer/docs/prototypes/ovstage-viewer.md">API and reproduction</a>.<br><strong>Implemented:</strong> external-scene rendering and CPU/CUDA pose transport. <strong>Proposed:</strong> the complete viewer upgrade, dependency changes and shared physics importer.</p>

## 1. Proposed architecture

USD (Universal Scene Description) composes geometry, materials, lights and environment into a scene. Preserve this scene; Newton imports physics and publishes motion through a body-to-prim mapping, which associates simulated bodies with scene objects. **Keep scene ownership and live USD prims outside `Model`; retain portable visuals for other viewers.**

| Component | Responsibility |
|---|---|
| **ovpopulation — load** | Uses OpenUSD to populate rendering and/or physics data in ovstage. Python: `ovstage.population.open_usd(...)`. |
| **ovstage — share** | Runtime scene store: prims, hierarchy, attributes and tensor access. Publication ordinals are increasing update identifiers that coordinate when changes become readable. |
| **ovnewton — connect** | Currently imports ovstage physics and synchronizes state. Proposed: a source reader plus bindings that reuse Newton’s import policy and pose transport. |

<!-- RUNTIME_FLOW -->

**ViewerRTX can remain independent of ovnewton.** Put shared pose transport in Newton’s optional ovstage integration; ovnewton delegates pose writes to it. OVRTX renders; ViewerRTX presents the result.

| Viewer mode | Scene ownership |
|---|---|
| **Existing USD / Isaac Lab scene** | Application owns stage, renderer and publication; ViewerRTX borrows them. |
| **Procedural Newton example** | Viewer creates its stage, geometry, camera and lights; uses the same transport and render path. |

## 2. OVRTX upgrade in Newton

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

## 3. Shared physics import and ownership

**One physics interpretation, two source readers.** Newton owns units, mass/inertia, collision geometry and joints. Readers supply resolved facts. Moving ovnewton’s parser while retaining Newton’s old parser would preserve the duplication.

<!-- IMPORT_FLOW -->

| Order / owner | Deliverable | Done when |
|---|---|---|
| **1 · Newton** | Complete the viewer upgrade above; use the existing USD importer during transition. | Both viewer modes pass headless/window, camera, overlay, deformation and cleanup checks on the qualified runtime pair. |
| **2 · Newton + ovnewton** | Newton exposes a source-neutral physics importer and remappable bindings. ovnewton supplies the ovstage reader, replaces its separate physics policy and reuses pose transport. | Readers produce equivalent models; bindings survive cloning, body collapse and reordering with correct affine offsets. |
| **3 · ovstage + ovnewton** | Expose authored/fallback/blocked distinctions; handle instanced colliders; align supported Newton and ovstage versions. | ANYmal, Franka and Cartpole import correctly with one jointly tested dependency set. |

<p class="caption">Steps 2–3 proceed together. Inspected ovnewton pins Newton <code>&gt;=1.5,&lt;1.6</code> and ovstage <code>0.2.0.378985</code>; it cannot simply be added to Newton 1.7 development. That alignment is separate from the viewer-only runtime pair above.</p>

**Mapping contract:** import bindings retain prim paths, stable body identities and rest offsets; resolve final indices after model construction. The prototype requires those indices explicitly. The application owns scene lifetime and publication.

## 4. Visual comparison

Both sides use OVRTX 0.5 with matched camera, lighting and authored settings. These are Isaac Lab asset-derived scenes, including a generated heightfield; **complete reinforcement-learning (RL) tasks and parity with Omniverse Kit rendering remain untested**. Select a scene and compare the overlay.

<!-- COMPARISONS -->

Preserving the source addresses appearance discarded during import in [#4051](https://github.com/newton-physics/newton/issues/4051). Exact Kit matching still depends on backend features, exposure, color management and temporal settings.

## 5. Performance and qualification

| Measured result | Required follow-up |
|---|---|
| **41–46 ms/frame** for 64 moving ANYmals, before physics | Profile publication before treating this as suitable for large RL workloads. |
| **20–21 ms** in global publication alone | ovstage/OVRTX team: reduce commit cost and retest moving scenes. |
| **+164–812 MiB GPU allocation** for small source scenes | Set workload-specific memory budgets; preserved appearance has a measurable cost. |
| Earlier import: **52.5 → 7.6 s** without visual extraction | Promising duplicate-work reduction; separate OVRTX 0.3 experiment, not a clone-pipeline comparison. |

<p class="caption">13 research runs · RTX 4090 / Windows · 640 × 480 · ≥30 s warmup + 60 frames. Synthetic pose replay, not solver execution; mostly one process repetition. GPU figures are allocation snapshots. Prototype: 64 robots at 41.4 ms/frame, including 22.5 ms encoding/writes/publication, without importing PXR or ovnewton.</p>

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
<li><a href="https://github.com/newton-physics/newton/pull/4170">PR #4170</a> and <a href="https://github.com/newton-physics/newton/pull/4177">PR #4177</a> can still benefit portable viewers; native RTX appearance should come from the source scene.</li>
</ul>
</details>
