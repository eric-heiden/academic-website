# ViewerGL Transparency Benchmark

| Scene | Mode | Mean ms | Median ms | Min ms | Max ms | Screenshot |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cloth_shirt_4 | sorted | 10.961 | 10.667 | 9.818 | 13.545 | cloth_shirt_4_sorted.png |
| cloth_shirt_4 | weighted_oit | 11.058 | 10.996 | 10.197 | 14.501 | cloth_shirt_4_weighted_oit.png |
| shape_cloud_240 | sorted | 3.638 | 3.444 | 3.089 | 5.491 | shape_cloud_240_sorted.png |
| shape_cloud_240 | weighted_oit | 2.941 | 2.830 | 2.247 | 3.788 | shape_cloud_240_weighted_oit.png |
| mesh_spheres_160 | sorted | 1.846 | 1.804 | 1.512 | 2.837 | mesh_spheres_160_sorted.png |
| mesh_spheres_160 | weighted_oit | 1.357 | 1.224 | 1.067 | 2.518 | mesh_spheres_160_weighted_oit.png |
| bunny_meshes_24 | sorted | 2.002 | 1.941 | 1.618 | 3.073 | bunny_meshes_24_sorted.png |
| bunny_meshes_24 | weighted_oit | 1.484 | 1.459 | 1.180 | 2.742 | bunny_meshes_24_weighted_oit.png |

## Relative Performance

| Scene | Sorted mean ms | Weighted OIT mean ms | OIT speedup |
| --- | ---: | ---: | ---: |
| cloth_shirt_4 | 10.961 | 11.058 | -0.9% |
| shape_cloud_240 | 3.638 | 2.941 | +23.7% |
| mesh_spheres_160 | 1.846 | 1.357 | +36.0% |
| bunny_meshes_24 | 2.002 | 1.484 | +34.9% |

## Image Differences

| Scene | Mean abs RGB | RMS RGB | Max abs RGB | Comparison |
| --- | ---: | ---: | ---: | --- |
| cloth_shirt_4 | 0.856 | 3.063 | 64 | cloth_shirt_4_compare.png |
| shape_cloud_240 | 1.846 | 6.295 | 71 | shape_cloud_240_compare.png |
| mesh_spheres_160 | 0.994 | 4.386 | 66 | mesh_spheres_160_compare.png |
| bunny_meshes_24 | 0.360 | 3.112 | 93 | bunny_meshes_24_compare.png |

## Scenes

- `cloth_shirt_4`: 4 transparent unisex_shirt.usd cloth meshes (12736 triangles each, 50944 rendered cloth triangles total).
- `shape_cloud_240`: 240 generated transparent primitive shapes plus an opaque ground plane.
- `mesh_spheres_160`: 160 generated transparent sphere mesh instances (1152 triangles each).
- `bunny_meshes_24`: 24 transparent bunny.usd mesh instances (12200 triangles each, 292800 mesh triangles total).
