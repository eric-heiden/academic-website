# ViewerGL Transparency Benchmark

| Scene | Mode | Mean ms | Median ms | Min ms | Max ms | Screenshot |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tri_surface_96 | sorted | 11.513 | 11.098 | 10.404 | 17.403 | tri_surface_96_sorted.png |
| tri_surface_96 | weighted_oit | 11.235 | 10.956 | 9.956 | 17.506 | tri_surface_96_weighted_oit.png |
| g1_multibody_1 | sorted | 6.721 | 6.596 | 5.896 | 8.418 | g1_multibody_1_sorted.png |
| g1_multibody_1 | weighted_oit | 6.219 | 6.183 | 5.435 | 7.793 | g1_multibody_1_weighted_oit.png |
| cloth_h1 | sorted | 22.368 | 21.971 | 20.494 | 27.553 | cloth_h1_sorted.png |
| cloth_h1 | weighted_oit | 23.543 | 23.010 | 20.674 | 28.343 | cloth_h1_weighted_oit.png |

## Image Differences

| Scene | Mean abs RGB | RMS RGB | Max abs RGB |
| --- | ---: | ---: | ---: |
| tri_surface_96 | 0.347 | 1.466 | 9 |
| g1_multibody_1 | 0.735 | 4.927 | 92 |
| cloth_h1 | 0.673 | 4.345 | 120 |

## Scenes

- `tri_surface_96`: 55296 transparent cloth triangles in crossing surfaces (uniform opacity).
- `g1_multibody_1`: 54 shapes from the Unitree G1 multibody asset; non-plane shapes opacity 0.45.
- `cloth_h1`: H1 jacket cloth reference scene with 66930 transparent cloth triangles and 99 transparent robot/ground shapes; opacity 0.45.
