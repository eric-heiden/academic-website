# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Grinding
#
# Removes workpiece volume by subtracting a moving grinding wheel from its
# texture SDF. Hydroelastic collision provides the pressure visualization;
# no dynamics solver is used.
#
# Command: python example_sdf_grinding.py
#
###########################################################################

import math

import newton
import newton.examples
import numpy as np
import warp as wp

WORKPIECE_RADII = (0.45, 0.25, 0.12)
WORKPIECE_RESOLUTION = 256
GRINDER_RADIUS = 0.13
GRINDER_HALF_WIDTH = 0.04
GRIND_DEPTH = 0.035
HYDROELASTIC_STIFFNESS = 1.0e8
PRESSURE_COLOR_MAX = 1.5e5

_SLOT_LINEAR = wp.uint32(0xFFFFFFFE)


@wp.func
def _sdf_cylinder_z(point: wp.vec3, radius: float, half_height: float) -> float:
    """Evaluate a capped cylinder aligned with its local Z-axis."""
    radial = wp.sqrt(point[0] * point[0] + point[1] * point[1]) - radius
    axial = wp.abs(point[2]) - half_height
    outside = wp.length(wp.vec2(wp.max(radial, 0.0), wp.max(axial, 0.0)))
    return outside + wp.min(wp.max(radial, axial), 0.0)


@wp.kernel(enable_backward=False)
def _subtract_cylinder_from_coarse_sdf(
    sdf_values: wp.array3d[wp.float32],
    sdf_lower: wp.vec3,
    voxel_size: wp.vec3,
    subgrid_size: int,
    grinder_xform: wp.transform,
    grinder_radius: float,
    grinder_half_width: float,
):
    z, y, x = wp.tid()
    grid_point = wp.vec3(float(x), float(y), float(z)) * float(subgrid_size)
    point = sdf_lower + wp.cw_mul(grid_point, voxel_size)
    grinder_point = wp.transform_point(wp.transform_inverse(grinder_xform), point)
    grinder_distance = _sdf_cylinder_z(
        grinder_point, grinder_radius, grinder_half_width
    )
    sdf_values[z, y, x] = wp.max(sdf_values[z, y, x], -grinder_distance)


@wp.kernel(enable_backward=False)
def _subtract_cylinder_from_subgrid_sdf(
    sdf_values: wp.array3d[wp.float32],
    subgrid_slots: wp.array3d[wp.uint32],
    sdf_lower: wp.vec3,
    voxel_size: wp.vec3,
    subgrid_size: int,
    grinder_xform: wp.transform,
    grinder_radius: float,
    grinder_half_width: float,
):
    tid = wp.tid()
    samples_per_dim = subgrid_size + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim
    block_id = tid // samples_per_subgrid
    sample_id = tid - block_id * samples_per_subgrid

    block_z = block_id // (subgrid_slots.shape[0] * subgrid_slots.shape[1])
    block_rem = block_id - block_z * subgrid_slots.shape[0] * subgrid_slots.shape[1]
    block_y = block_rem // subgrid_slots.shape[0]
    block_x = block_rem - block_y * subgrid_slots.shape[0]

    local_z = sample_id // (samples_per_dim * samples_per_dim)
    sample_rem = sample_id - local_z * samples_per_dim * samples_per_dim
    local_y = sample_rem // samples_per_dim
    local_x = sample_rem - local_y * samples_per_dim

    slot = subgrid_slots[block_x, block_y, block_z]
    if slot >= _SLOT_LINEAR:
        return
    address_x = int(slot & wp.uint32(0x3FF))
    address_y = int((slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    address_z = int((slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    texture_x = address_x * samples_per_dim + local_x
    texture_y = address_y * samples_per_dim + local_y
    texture_z = address_z * samples_per_dim + local_z

    fine_x = block_x * subgrid_size + local_x
    fine_y = block_y * subgrid_size + local_y
    fine_z = block_z * subgrid_size + local_z
    point = sdf_lower + wp.cw_mul(
        wp.vec3(float(fine_x), float(fine_y), float(fine_z)), voxel_size
    )
    grinder_point = wp.transform_point(wp.transform_inverse(grinder_xform), point)
    grinder_distance = _sdf_cylinder_z(
        grinder_point, grinder_radius, grinder_half_width
    )
    sdf_values[texture_z, texture_y, texture_x] = wp.max(
        sdf_values[texture_z, texture_y, texture_x], -grinder_distance
    )


@wp.func
def _pressure_color(pressure: float, max_pressure: float) -> wp.vec3:
    """Map pressure from blue through cyan and yellow to red."""
    t = wp.clamp(pressure / max_pressure, 0.0, 1.0)
    if t < 0.25:
        return wp.vec3(0.0, t * 4.0, 1.0)
    if t < 0.5:
        return wp.vec3(0.0, 1.0, 2.0 - t * 4.0)
    if t < 0.75:
        return wp.vec3(t * 4.0 - 2.0, 1.0, 0.0)
    return wp.vec3(1.0, 4.0 - t * 4.0, 0.0)


@wp.kernel(enable_backward=False)
def _build_pressure_lines(
    triangle_vertices: wp.array[wp.vec3],
    face_depths: wp.array[wp.float32],
    face_pressures: wp.array[wp.float32],
    max_pressure: float,
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
    line_colors: wp.array[wp.vec3],
):
    face = wp.tid()
    base = face * 3
    if face_depths[face] >= 0.0:
        for edge in range(3):
            line_starts[base + edge] = wp.vec3(0.0)
            line_ends[base + edge] = wp.vec3(0.0)
            line_colors[base + edge] = wp.vec3(0.0)
        return

    color = _pressure_color(face_pressures[face], max_pressure)
    for edge in range(3):
        line_starts[base + edge] = triangle_vertices[base + edge]
        line_ends[base + edge] = triangle_vertices[base + (edge + 1) % 3]
        line_colors[base + edge] = color


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.frame = 0
        self.test_mode = args.test

        # A gently curved blank keeps fine SDF subgrids resident around the
        # entire machined surface, which makes this compact in-place demo
        # independent of Newton's linear-subgrid storage optimization.
        workpiece_mesh = newton.Mesh.create_ellipsoid(
            *WORKPIECE_RADII,
            num_latitudes=32,
            num_longitudes=64,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.workpiece_sdf = workpiece_mesh.build_sdf(
            max_resolution=WORKPIECE_RESOLUTION,
            narrow_band_range=(-0.08, 0.08),
            margin=0.04,
            texture_format="float32",
            paired_samples=False,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.005,
            kh=HYDROELASTIC_STIFFNESS,
            density=0.0,
            is_hydroelastic=True,
        )
        workpiece_cfg = shape_cfg.copy()
        workpiece_cfg.is_visible = False
        grinder_cfg = shape_cfg.copy()
        grinder_cfg.sdf_max_resolution = 64
        grinder_cfg.sdf_narrow_band_range = (-0.06, 0.06)
        grinder_cfg.sdf_padding = 0.01

        builder = newton.ModelBuilder()
        builder.sdf_texture_paired_samples = False
        builder.add_shape_mesh(
            body=-1,
            mesh=workpiece_mesh,
            cfg=workpiece_cfg,
            label="workpiece",
        )

        initial_pose = self._grinder_pose(0)
        self.grinder_body = builder.add_body(xform=initial_pose, label="grinder")
        builder.add_shape_cylinder(
            body=self.grinder_body,
            radius=GRINDER_RADIUS,
            half_height=GRINDER_HALF_WIDTH,
            cfg=grinder_cfg,
            color=(0.18, 0.20, 0.23),
            opacity=0.4,
            label="grinding_wheel",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.contacts = None

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=False,
            sdf_hydroelastic_config=newton.geometry.HydroelasticSDF.Config(
                mc_edge_clamp_min=0.0,
                output_contact_surface=True,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()

        texture_data = self.workpiece_sdf.texture_data
        if texture_data is None:
            raise RuntimeError("The workpiece did not produce a texture SDF.")
        self.texture_data = texture_data
        self.coarse_values = wp.empty(
            (
                texture_data.coarse_texture.depth,
                texture_data.coarse_texture.height,
                texture_data.coarse_texture.width,
            ),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.subgrid_values = wp.empty(
            (
                texture_data.subgrid_texture.depth,
                texture_data.subgrid_texture.height,
                texture_data.subgrid_texture.width,
            ),
            dtype=wp.float32,
            device=self.model.device,
        )
        texture_data.coarse_texture.copy_to(self.coarse_values)
        texture_data.subgrid_texture.copy_to(self.subgrid_values)
        self.initial_subgrid_values = (
            self.subgrid_values.numpy() if self.test_mode else None
        )

        self.subgrid_work_items = (
            texture_data.subgrid_start_slots.size
            * (int(texture_data.subgrid_size) + 1) ** 3
        )

        self.body_q = self.state_0.body_q.numpy()
        self.max_pressure = 0.0
        self.latest_pressure = 0.0
        self.pressure_line_starts = None
        self.pressure_line_ends = None
        self.pressure_line_colors = None
        self._set_grinder_pose(initial_pose)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._track_pressure()
        self.initial_pressure = self.latest_pressure
        self._update_workpiece_surface()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.8, -0.85, 0.5), pitch=-18.0, yaw=132.0)

    @staticmethod
    def _grinder_pose(frame: int) -> wp.transform:
        x_start = -0.36
        x_end = 0.36
        x = min(x_start + frame * 0.004, x_end)
        radial_fraction = min((x / WORKPIECE_RADII[0]) ** 2, 1.0)
        surface_z = WORKPIECE_RADII[2] * math.sqrt(1.0 - radial_fraction)
        z = surface_z + GRINDER_RADIUS - GRIND_DEPTH
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        return wp.transform(wp.vec3(x, 0.0, z), rotation)

    def _set_grinder_pose(self, pose: wp.transform) -> None:
        self.body_q[self.grinder_body] = np.asarray(pose)
        self.state_0.body_q.assign(self.body_q)

    def _subtract_grinder(self, pose: wp.transform) -> None:
        wp.launch(
            _subtract_cylinder_from_coarse_sdf,
            dim=self.coarse_values.shape,
            inputs=[
                self.coarse_values,
                self.texture_data.sdf_box_lower,
                self.texture_data.voxel_size,
                int(self.texture_data.subgrid_size),
                pose,
                GRINDER_RADIUS,
                GRINDER_HALF_WIDTH,
            ],
            device=self.model.device,
        )
        wp.launch(
            _subtract_cylinder_from_subgrid_sdf,
            dim=self.subgrid_work_items,
            inputs=[
                self.subgrid_values,
                self.texture_data.subgrid_start_slots,
                self.texture_data.sdf_box_lower,
                self.texture_data.voxel_size,
                int(self.texture_data.subgrid_size),
                pose,
                GRINDER_RADIUS,
                GRINDER_HALF_WIDTH,
            ],
            device=self.model.device,
        )
        self.texture_data.coarse_texture.copy_from(self.coarse_values)
        self.texture_data.subgrid_texture.copy_from(self.subgrid_values)

    def _track_pressure(self) -> None:
        if not self.test_mode:
            return
        hydroelastic = self.collision_pipeline.hydroelastic_sdf
        contact_surface = (
            hydroelastic.get_contact_surface() if hydroelastic is not None else None
        )
        if contact_surface is None:
            return
        self.latest_pressure = 0.0
        face_count = min(
            int(contact_surface.face_contact_count.numpy()[0]),
            contact_surface.max_num_face_contacts,
        )
        if face_count > 0:
            pressures = hydroelastic.contact_reduction.reducer.contact_pressure.numpy()[
                :face_count
            ]
            self.latest_pressure = float(np.max(pressures))
            self.max_pressure = max(self.max_pressure, self.latest_pressure)

    def _log_pressure_surface(self) -> None:
        hydroelastic = self.collision_pipeline.hydroelastic_sdf
        contact_surface = (
            hydroelastic.get_contact_surface() if hydroelastic is not None else None
        )
        if contact_surface is None:
            self.viewer.log_lines("/hydro_pressure", None, None, None)
            return

        face_count = min(
            int(contact_surface.face_contact_count.numpy()[0]),
            contact_surface.max_num_face_contacts,
        )
        if face_count == 0:
            self.viewer.log_lines("/hydro_pressure", None, None, None)
            return

        max_lines = 3 * contact_surface.max_num_face_contacts
        if (
            self.pressure_line_starts is None
            or len(self.pressure_line_starts) < max_lines
        ):
            self.pressure_line_starts = wp.empty(
                max_lines, dtype=wp.vec3, device=self.model.device
            )
            self.pressure_line_ends = wp.empty(
                max_lines, dtype=wp.vec3, device=self.model.device
            )
            self.pressure_line_colors = wp.empty(
                max_lines, dtype=wp.vec3, device=self.model.device
            )

        wp.launch(
            _build_pressure_lines,
            dim=face_count,
            inputs=[
                contact_surface.contact_surface_point,
                contact_surface.contact_surface_depth,
                hydroelastic.contact_reduction.reducer.contact_pressure,
                PRESSURE_COLOR_MAX,
            ],
            outputs=[
                self.pressure_line_starts,
                self.pressure_line_ends,
                self.pressure_line_colors,
            ],
            device=self.model.device,
        )
        num_lines = 3 * face_count
        self.viewer.log_lines(
            "/hydro_pressure",
            self.pressure_line_starts[:num_lines],
            self.pressure_line_ends[:num_lines],
            self.pressure_line_colors[:num_lines],
        )

    def _update_workpiece_surface(self) -> None:
        surface = self.workpiece_sdf.extract_isomesh(device=self.model.device)
        if surface is None:
            raise RuntimeError("Grinding removed the complete workpiece.")
        self.workpiece_points = wp.array(
            surface.vertices, dtype=wp.vec3, device=self.model.device
        )
        self.workpiece_indices = wp.array(
            surface.indices.reshape(-1), dtype=wp.int32, device=self.model.device
        )

    def step(self):
        self.frame += 1
        grinder_pose = self._grinder_pose(self.frame)
        self._set_grinder_pose(grinder_pose)

        # Measure the hydroelastic pressure on the current surface, then remove
        # the wheel volume from the workpiece with an SDF CSG difference.
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._track_pressure()
        self._subtract_grinder(grinder_pose)
        self._update_workpiece_surface()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/workpiece_sdf",
            self.workpiece_points,
            self.workpiece_indices,
            color=(0.55, 0.62, 0.68),
            roughness=0.65,
            metallic=0.25,
            dynamic=True,
        )
        self._log_pressure_surface()
        self.viewer.end_frame()

    def test_final(self):
        """Verify grinding changes the SDF and produces hydroelastic pressure."""
        assert self.initial_subgrid_values is not None
        final_values = self.subgrid_values.numpy()
        max_removed_distance = float(np.max(final_values - self.initial_subgrid_values))
        assert max_removed_distance > 0.01, (
            f"Workpiece SDF changed by only {max_removed_distance:.6f} m."
        )
        assert self.max_pressure > 0.0, (
            "Grinding produced no positive hydroelastic pressure."
        )

        # Re-collide after the final CSG update to verify that hydroelastic
        # sampling observes the edited texture through the original handle.
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._track_pressure()
        assert self.latest_pressure < 0.5 * self.initial_pressure, (
            "Hydroelastic collision did not observe the removed SDF volume."
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
