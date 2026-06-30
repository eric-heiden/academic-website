# ViewerGL Transparency Benchmark

| Scene | Mode | Mean ms | Median ms | Min ms | Max ms | Screenshot |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cloth_shirt_4 | sorted | 13.541 | 13.493 | 11.948 | 15.510 | cloth_shirt_4_sorted.png |
| cloth_shirt_4 | weighted_oit | 16.152 | 15.847 | 14.727 | 18.578 | cloth_shirt_4_weighted_oit.png |
| shape_cloud_240 | sorted | 9.801 | 9.472 | 8.793 | 13.183 | shape_cloud_240_sorted.png |
| shape_cloud_240 | weighted_oit | 12.519 | 12.076 | 11.354 | 15.767 | shape_cloud_240_weighted_oit.png |
| mesh_spheres_160 | sorted | 8.889 | 8.552 | 7.921 | 11.908 | mesh_spheres_160_sorted.png |
| mesh_spheres_160 | weighted_oit | 12.313 | 12.246 | 11.011 | 14.117 | mesh_spheres_160_weighted_oit.png |
| bunny_meshes_24 | sorted | 8.190 | 7.880 | 7.260 | 11.644 | bunny_meshes_24_sorted.png |
| bunny_meshes_24 | weighted_oit | 11.386 | 11.215 | 9.848 | 14.229 | bunny_meshes_24_weighted_oit.png |

## Relative Performance

| Scene | Sorted mean ms | Weighted OIT mean ms | OIT speedup |
| --- | ---: | ---: | ---: |
| cloth_shirt_4 | 13.541 | 16.152 | -16.2% |
| shape_cloud_240 | 9.801 | 12.519 | -21.7% |
| mesh_spheres_160 | 8.889 | 12.313 | -27.8% |
| bunny_meshes_24 | 8.190 | 11.386 | -28.1% |

## Image Differences

| Scene | Mean abs RGB | RMS RGB | Max abs RGB | Comparison |
| --- | ---: | ---: | ---: | --- |
| cloth_shirt_4 | 0.839 | 3.049 | 66 | cloth_shirt_4_compare.png |
| shape_cloud_240 | 1.815 | 6.264 | 72 | shape_cloud_240_compare.png |
| mesh_spheres_160 | 0.971 | 4.367 | 66 | mesh_spheres_160_compare.png |
| bunny_meshes_24 | 0.356 | 3.101 | 98 | bunny_meshes_24_compare.png |

## Scenes

- `cloth_shirt_4`: 4 transparent unisex_shirt.usd cloth meshes (12736 triangles each, 50944 rendered cloth triangles total).
- `shape_cloud_240`: 240 generated transparent primitive shapes plus an opaque ground plane.
- `mesh_spheres_160`: 160 generated transparent sphere mesh instances (1152 triangles each).
- `bunny_meshes_24`: 24 transparent bunny.usd mesh instances (12200 triangles each, 292800 mesh triangles total).
