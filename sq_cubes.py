#!/usr/bin/env python
# coding: utf-8
"""
Structure factor S(q) of hard cubes from HOOMD-blue Monte Carlo.

Protocol:
  1. Generate a dilute lattice of hard cubes.
  2. Compress the box to a target packing fraction phi = N Vp / V.
  3. Equilibrate with HPMC (translation + rotation moves).
  4. Compute S(q) with freud.

The cube side length is 1 (V_p = 1), so phi = N / L^3.
The result is plotted vs dimensionless q*L.

Requires: HOOMD-blue, freud, gsd. Optional: fresnel (rendering).
"""

import os
import math
import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd
import hoomd
import hoomd.hpmc
import freud

try:
    import fresnel
    import IPython
    _HAS_FRESNEL = True
except ImportError:
    _HAS_FRESNEL = False


# ----------------------------------------------------------------------
# Simulation setup
# ----------------------------------------------------------------------
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=12)
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
simulation.operations.integrator = mc


# ----------------------------------------------------------------------
# Particle shape: unit cube (vertices at +/- 0.5)
# ----------------------------------------------------------------------
CUBE_VERTICES = [
    (-0.5, -0.5, -0.5),
    (-0.5, -0.5,  0.5),
    (-0.5,  0.5, -0.5),
    (-0.5,  0.5,  0.5),
    ( 0.5, -0.5, -0.5),
    ( 0.5, -0.5,  0.5),
    ( 0.5,  0.5, -0.5),
    ( 0.5,  0.5,  0.5),
]
mc.shape["cube"] = dict(vertices=CUBE_VERTICES)
mc.nselect = 2
mc.d["cube"] = 0.15
mc.a["cube"] = 0.2


# ----------------------------------------------------------------------
# Optional: Fresnel renderer
# ----------------------------------------------------------------------
def render(snapshot, w=300, h=300, samples=2000):
    """Render a snapshot with Fresnel. Requires the fresnel package."""
    if not _HAS_FRESNEL:
        warnings.warn("fresnel is not installed; skipping rendering.")
        return None

    device = fresnel.Device()
    tracer = fresnel.tracer.Path(device=device, w=w, h=h)

    L = snapshot.configuration.box[0]
    poly_info = fresnel.util.convex_polyhedron_from_vertices(CUBE_VERTICES)

    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.ConvexPolyhedron(
        scene, poly_info, N=snapshot.particles.N
    )
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([0.01, 0.74, 0.26]), roughness=0.5
    )
    geometry.position[:] = snapshot.particles.position[:]
    geometry.orientation[:] = snapshot.particles.orientation[:]
    geometry.outline_width = 0.01
    fresnel.geometry.Box(scene, snapshot.configuration.box, box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8),
                            theta=math.pi),
        fresnel.light.Light(direction=(1, 1, 1), color=(1.1, 1.1, 1.1),
                            theta=math.pi / 3),
    ]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        height=L * 1.4 + 1,
    )
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)

    if "CI" in os.environ:
        samples = 100
    return IPython.display.Image(tracer.sample(scene, samples=samples)._repr_png_())


# ----------------------------------------------------------------------
# Initial dilute lattice
# ----------------------------------------------------------------------
m = 16
N_particles = 2 * m ** 3
spacing = 1.2
K = math.ceil(N_particles ** (1 / 3))
L = K * spacing

x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))[:N_particles]
orientation = [(1, 0, 0, 0)] * N_particles

frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position
frame.particles.orientation = orientation
frame.particles.typeid = [0] * N_particles
frame.particles.types = ["cube"]
frame.configuration.box = [L, L, L, 0, 0, 0]

# Clean up any previous lattice file
lattice_fn = os.path.join(os.getcwd(), "lattice.gsd")
if os.path.exists(lattice_fn):
    os.remove(lattice_fn)

with gsd.hoomd.open(name=lattice_fn, mode="x") as f:
    f.append(frame)
simulation.create_state_from_gsd(filename=lattice_fn)

if _HAS_FRESNEL:
    render(simulation.state.get_snapshot())

simulation.run(2e3)
if _HAS_FRESNEL:
    render(simulation.state.get_snapshot())


# ----------------------------------------------------------------------
# Compression to target packing fraction
# ----------------------------------------------------------------------
V_particle = 1.0
final_phi = 0.365
initial_box = simulation.state.box
final_box = hoomd.Box.from_box(initial_box)
final_box.volume = simulation.state.N_particles * V_particle / final_phi

compress = hoomd.hpmc.update.QuickCompress(
    trigger=hoomd.trigger.Periodic(10), target_box=final_box
)
simulation.operations.updaters.append(compress)

periodic = hoomd.trigger.Periodic(10)
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=["a", "d"],
    target=0.2,
    trigger=periodic,
    max_translation_move=0.2,
    max_rotation_move=0.2,
)
simulation.operations.tuners.append(tune)

while not compress.complete and simulation.timestep < 1e6:
    simulation.run(1000)
if not compress.complete:
    raise RuntimeError("Compression failed to complete")

if _HAS_FRESNEL:
    render(simulation.state.get_snapshot())

# Short tuning pass
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=["a", "d"],
    target=0.2,
    trigger=hoomd.trigger.And([
        hoomd.trigger.Periodic(100),
        hoomd.trigger.Before(simulation.timestep + 5000),
    ]),
)
simulation.operations.tuners.append(tune)
simulation.run(5000)

# ----------------------------------------------------------------------
# Equilibration
# ----------------------------------------------------------------------
simulation.run(2e3)

equilibrated_fn = os.path.join(os.getcwd(), "equilibrated.gsd")
hoomd.write.GSD.write(state=simulation.state, filename=equilibrated_fn)

if _HAS_FRESNEL:
    img = render(simulation.state.get_snapshot())
    if img is not None:
        with open("render.png", "wb") as f:
            f.write(img.data)


# ----------------------------------------------------------------------
# Structure factor S(q) with freud
# ----------------------------------------------------------------------
bins = 300
k_max = 40
k_min = 1

sf_direct = freud.diffraction.StaticStructureFactorDirect(
    bins=bins, k_max=k_max, k_min=k_min
)
sf_debye = freud.diffraction.StaticStructureFactorDebye(
    num_k_values=bins, k_max=k_max, k_min=k_min
)

with gsd.hoomd.open(equilibrated_fn, "r") as traj:
    for frame in traj:
        sf_direct.compute(frame, reset=False)
        sf_debye.compute(frame, reset=False)


# ----------------------------------------------------------------------
# Plot S(q) vs qL
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sf_direct.bin_centers, sf_direct.S_k, label="Direct")
ax.set_title("Static Structure Factor")
ax.set_xlabel(r"$qL$")
ax.set_ylabel(r"$S(q)$")
ax.set_ylim(bottom=0)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
