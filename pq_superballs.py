#!/usr/bin/env python
# coding: utf-8
"""
Form factor P(q) of a superball from numerical FFT.

A superball is defined by:

    |x/D|^(2p) + |y/D|^(2p) + |z/D|^(2p) <= 1

    p = 1    -> sphere
    p = 1.5  -> rounded cube
    p -> inf -> cube

The form factor is computed by FFT of the shape on a regular 3D grid and
radial binning of |FFT|^2. The result is plotted vs the dimensionless q*D.

Built on numpy + scipy. No micromagnetic solver required.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import binned_statistic
from scipy.integrate import simpson
from tqdm import tqdm


# ----------------------------------------------------------------------
# Physical and numerical parameters
# ----------------------------------------------------------------------
muB = 9.274009994e-24       # J/T (only used to fix Ms, not for P(q))
NA = 6.022e23
rho = 5170                  # kg/m^3
MW = 234.15                 # g/mol
Ms = 2 * muB * NA * rho * 1000 / MW

n0 = 0.838e-9               # real-space grid spacing (m)
dist = 10 * n0              # minimum gap between superballs (m)
Rsph = 14 * n0              # superball "radius" (m)
dNP = Rsph + dist           # nominal particle size (m)
a = 1                       # particles per dimension
d = a * dNP

p = 1.5                     # superball exponent (p=1 sphere, p->inf cube)
theta = np.pi / 2           # rotation around y (rad)
phi = 3 * np.pi / 4         # rotation around z (rad)

pad_factor = 1              # zero-padding factor for the FFT
bins = 300                  # number of radial bins

# Simulation box: 6 * d plus a margin
L = 6 * d + 2 * dNP

# Reproducibility (optional)
np.random.seed(42)


# ----------------------------------------------------------------------
# 3D grid
# ----------------------------------------------------------------------
Nx = int(np.ceil(L / n0))
x = np.linspace(-L / 2, L / 2, Nx, endpoint=False, dtype=np.float32)
y = x.copy()
z = x.copy()
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# ----------------------------------------------------------------------
# Superball shape (rotated)
# ----------------------------------------------------------------------
def superball_rotated(x, y, z, theta=np.pi / 3, phi=np.pi / 6, p=p):
    """
    Return Ms inside a rotated superball centered at the origin, 0 outside.

    Parameters
    ----------
    x, y, z : ndarray
        Coordinates (m).
    theta : float
        Rotation around y (rad).
    phi : float
        Rotation around z (rad).
    p : float
        Superball exponent (p=1 sphere, p->inf cube).
    """
    Rz = np.array([
        [ np.cos(phi), -np.sin(phi), 0],
        [ np.sin(phi),  np.cos(phi), 0],
        [           0,            0, 1],
    ], dtype=np.float32)
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)],
    ], dtype=np.float32)
    R = Ry @ Rz

    coords = np.stack((x, y, z), axis=-1)
    coords_rot = coords @ R.T
    xr, yr, zr = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]

    scale = (dNP - dist) / 2.0
    x_rel = xr / scale
    y_rel = yr / scale
    z_rel = zr / scale

    inside = (
        np.abs(x_rel) ** (2 * p)
        + np.abs(y_rel) ** (2 * p)
        + np.abs(z_rel) ** (2 * p)
    ) <= 1

    return np.where(inside, Ms, 0).astype(np.float32)


# ----------------------------------------------------------------------
# Build the shape on the grid
# ----------------------------------------------------------------------
start = time.time()
magnitude = superball_rotated(X, Y, Z, theta=theta, phi=phi, p=p)
del X, Y, Z
print(f"Time to build the shape: {time.time() - start:.2f} s")

# Only the z-component is non-zero: the shape is a scalar field
m_z = magnitude.astype(np.float32)
del magnitude


# ----------------------------------------------------------------------
# Zero-padding for higher FFT resolution
# ----------------------------------------------------------------------
pad = Nx * (pad_factor - 1) // 2
pad_width = ((pad, pad),) * 3
m_z_padded = np.pad(m_z, pad_width, mode='constant', constant_values=0)
m_z_padded = m_z_padded.astype(np.float32)


# ----------------------------------------------------------------------
# 2D visualization of the central plane
# ----------------------------------------------------------------------
nonzero_counts = [np.count_nonzero(m_z_padded[:, :, i]) for i in range(m_z_padded.shape[2])]
z_index = int(np.argmax(nonzero_counts))
central_slice = m_z_padded[:, :, z_index]

extent_nm = [x.min() * 1e9, x.max() * 1e9, y.min() * 1e9, y.max() * 1e9]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(central_slice.T, origin='lower', cmap='inferno', extent=extent_nm)
fig.colorbar(im, ax=ax, label='Ms')
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title(rf'$\theta={theta:.2f},\ \phi={phi:.2f}$')
plt.tight_layout()
plt.show()

del central_slice, extent_nm


# ----------------------------------------------------------------------
# 3D FFT and radial binning
# ----------------------------------------------------------------------
start = time.time()
fft_mz = np.fft.fftshift(np.fft.fftn(m_z_padded)).astype(np.complex64)
psd = (np.abs(fft_mz) ** 2).astype(np.float32)
del fft_mz
print(f"Time for 3D FFT: {time.time() - start:.2f} s")

N_pad = m_z_padded.shape[0]

# Take the kx = 0 plane
psd_2D = psd[N_pad // 2, :, :].copy()
del psd, m_z

ky = np.fft.fftshift(np.fft.fftfreq(N_pad, d=n0)) * 2 * np.pi
kz = ky.copy()
KY_2D, KZ_2D = np.meshgrid(ky, kz, indexing='ij')

k_radial = np.sqrt(KY_2D ** 2 + KZ_2D ** 2).ravel()
psd_flat = psd_2D.ravel()

# Radial average to get P(q)
I_mean, bin_edges, _ = binned_statistic(
    k_radial, psd_flat, statistic='mean', bins=bins
)
q_mean = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Normalize so that P(0) = 1
I_norm = I_mean / np.nanmax(I_mean)

del psd_flat, psd_2D, k_radial, m_z_padded


# ----------------------------------------------------------------------
# Plot P(q) vs qD
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(q_mean * 1e-9, I_norm, color='black', label='P(q)')
ax.set_xlabel(r'$q$ (nm$^{-1}$)')
ax.set_ylabel(r'$I(q)/I(0)$')
ax.set_xlim(0.1, 2.5)
ax.set_yscale('log')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
