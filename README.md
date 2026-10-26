# SAXS Form and Structure Factors: Superballs and Cubes

Python tools to compute the **form factor** `P(q)` of **superballs** and the
**structure factor** `S(q)` of **hard cubes**, for modeling the Small-Angle
X-ray Scattering (SAXS) intensity of nanoparticle assemblies.

Both codes solve the Fourier transform **numerically**, which has a key
advantage: the `q` axis is rendered dimensionless by multiplying by the
particle size `D`. The curves are then **general** — they can be compared
directly with SAXS data from any particle size, once `P(q)·S(q)` is scaled
appropriately.

The two codes are complementary:

- `pq_superballs.py`  →  form factor `P(q)` of a superball
- `sq_cubes.py`       →  structure factor `S(q)` of a hard-cube fluid

The total SAXS intensity is `I(q) = P(q) · S(q)` (in the decoupling
approximation).

---

## Physics

### Superballs (form factor)

A superball is the family of shapes defined by

    |x/D|^(2p) + |y/D|^(2p) + |z/D|^(2p) ≤ 1

with `2p` the exponent (the "squareness" parameter):

| Exponent | Shape |
|---|---|
| `p = 1` | sphere |
| `p = 1.5` | rounded cube (used here) |
| `p → ∞` (e.g. `p = 50`) | cube |

The form factor is the squared modulus of the Fourier transform of the
shape, normalized by the particle volume:

    P(q) = | ∫_V e^{i q·r} d³r |² / V²

It is computed numerically by:

1. placing the superball on a regular 3D grid,
2. taking a 3D FFT (with optional zero-padding) of the shape,
3. averaging `|FFT|²` over spherical shells of constant `|q|`.

The output is `P(q)` as a function of `qD` (dimensionless).

### Hard cubes (structure factor)

A fluid of hard cubes is simulated with **HOOMD-blue** using the
Hard Particle Monte Carlo (HPMC) integrator. The protocol is:

1. **Generation**: a dilute lattice of cubes is placed in a cubic box.
2. **Expansion / compression**: the box is slowly compressed
   (QuickCompress) to reach the target **packing fraction**

       φ = N · V_p / V

   where `N` is the number of particles, `V_p = 1` the volume of one cube
   (side length 1), and `V = L³` the box volume.
3. **Equilibration**: MC moves (translation + rotation) are tuned to a
   target acceptance of ~20 %.

The structure factor is then computed from the equilibrated configuration
using `freud.diffraction.StaticStructureFactorDirect` (or `Debye`):

    S(q) = (1/N) ⟨ Σ_{i,j} e^{i q·(r_i - r_j)} ⟩

The output is `S(q)` as a function of `qD` (dimensionless, with `D` the cube
side length).

---

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`
- For `sq_cubes.py`:
  - [HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue/) ≥ 4.0
  - [freud](https://freud.readthedocs.io/)
  - [gsd](https://gsd.readthedocs.io/)
  - [fresnel](https://fresnel.readthedocs.io/) (optional, only for rendering)

Install:

```bash
pip install -r requirements.txt
