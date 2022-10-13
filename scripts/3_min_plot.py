"""Create plots for the 3-minute UROP Competition."""

import numpy as np
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Setting parameters
E = 2
e = 1

H = 1
L = E / H
l = e / H  # noqa: E741
delta = e / E

# Construct the differentiation matrices
n = 20
block_len = (n + 1) ** 2
D, c = cheb(n)

eps = c[1]

# Construct the grid of points
x = np.concatenate(
    ((l / 4) * (c / eps - 1), ((L - l) * c / eps - (L + l)) / 4)
)
y = (c / eps + 1) / 2

xx_std, yy_std = np.meshgrid(x, y)

xx = np.vstack((xx_std[:, : n + 1], xx_std[:, n + 1 :]))
yy = np.vstack((yy_std[:, : n + 1], yy_std[:, n + 1 :]))

xf = xx.flatten()
yf = yy.flatten()

interior = (
    (xf <= x[1])
    & (xf >= x[-2])
    & (yf <= y[1])
    & (yf >= y[-2])
    & (xf != x[n])
    & (xf != x[n + 1])
)

# Construct the relevant operators
Dx = np.kron(np.eye(n + 1), D)
Dy = np.kron(D, np.eye(n + 1))

Dxy = Dx @ Dy

D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))

D2x2y = D2x @ D2y

D3 = D2 @ D
D3x = np.kron(np.eye(n + 1), D3)

D4 = D2 @ D2
D4x = np.kron(np.eye(n + 1), D4)
D4y = np.kron(D4, np.eye(n + 1))

L1 = np.zeros((block_len, block_len))
L2 = np.zeros((block_len, block_len))

A1 = np.zeros((block_len, block_len))
A2 = np.zeros((block_len, block_len))

f1 = np.zeros(block_len)
f2 = f1.copy()

# Populate matrices for interior points
L1 = (4 / l) ** 4 * D4x + 2 * (4 / l) ** 2 * 2**2 * D2x2y + 2**4 * D4y
L2 = (
    (4 / (L - l)) ** 4 * D4x
    + 2 * (4 / (L - l)) ** 2 * 2**2 * D2x2y
    + 2**4 * D4y
)

# Symmetry conditions
L1[-2 * (n + 1) : -(n + 1), :] = D2y[
    -2 * (n + 1) : -(n + 1), :
]  # Symmetry y = 0, eta = -1
A1[-2 * (n + 1) : -(n + 1), :] = 0
f1[-2 * (n + 1) : -(n + 1)] = 0

L1[-(n + 1) :, :] = Dx[-2 * (n + 1) : -(n + 1), :]  # No-flow y = 0, eta = -1
A1[-(n + 1) :, :] = 0
f1[-(n + 1) :] = 0

L1[1 :: n + 1, :] = D2x[1 :: n + 1, :]  # Symmetry x = 0, xi = 1
A1[1 :: n + 1, :] = 0
f1[1 :: n + 1] = 0

L1[:: n + 1, :] = Dx[1 :: n + 1, :]  # Symmetry x = 0, xi = 1
A1[:: n + 1, :] = 0
f1[:: n + 1] = 0

L2[-2 * (n + 1) : -(n + 1), :] = D2y[
    -2 * (n + 1) : -(n + 1), :
]  # Symmetry y = 0, eta = -1
A2[-2 * (n + 1) : -(n + 1), :] = 0
f2[-2 * (n + 1) : -(n + 1)] = 0

L2[-(n + 1) :, :] = Dx[-2 * (n + 1) : -(n + 1), :]  # No-flow y = 0, eta = -1
A2[-(n + 1) :, :] = 0
f2[-(n + 1) :] = 0

L2[n - 1 :: n + 1, :] = D2x[n - 1 :: n + 1, :]  # Symmetry x = -L/2, xi = -1
A2[n - 1 :: n + 1, :] = 0
f2[n - 1 :: n + 1] = 0

L2[n :: n + 1, :] = Dx[n - 1 :: n + 1, :]  # Symmetry x = -L/2, xi = -1
A2[n :: n + 1, :] = 0
f2[n :: n + 1] = 0

# Interlace the domains along the shared boundary
L1[n - 1 :: n + 1, :] = (1 / l) ** 3 * D3x[n - 1 :: n + 1, :]
A1[n - 1 :: n + 1, :] = -((1 / (L - l)) ** 3) * D3x[1 :: n + 1, :]
f1[n - 1 :: n + 1] = 0

L1[n :: n + 1, :] = (1 / l) * Dx[n - 1 :: n + 1, :]
A1[n :: n + 1, :] = -(1 / (L - l)) * Dx[1 :: n + 1, :]
f1[n :: n + 1] = 0

L2[:: n + 1, :] = (1 / (L - l)) ** 2 * D2x[1 :: n + 1, :]
A2[:: n + 1, :] = -((1 / l) ** 2) * D2x[n - 1 :: n + 1, :]
f2[:: n + 1] = 0

# Top boundary conditions
L1[: n + 1, :] = D2y[n + 1 : 2 * (n + 1), :]  # Interface y = 1, eta = 1
A1[: n + 1, :] = 0
f1[: n + 1] = 1 / (4 * eps**2)

L2[: n + 1, :] = Dy[n + 1 : 2 * (n + 1), :]  # No-flow y = 1, eta = 1
A2[: n + 1, :] = 0
f2[: n + 1] = 0

# Final interlacing conditions
L2[1 :: n + 1, :] = 0
A2[1 :: n + 1, :] = 0
L2[1 :: n + 1, 1 :: n + 1] = np.eye(n + 1)
A2[1 :: n + 1, n - 1 :: n + 1] = -np.eye(n + 1)
f2[1 :: n + 1] = 0

# Streamfunction condition
L1[n + 1 : 2 * (n + 1), :] = 0  # Streamfunction y = 1, eta = 1
A1[n + 1 : 2 * (n + 1), :] = 0
L1[n + 1 : 2 * (n + 1), n + 1 : 2 * (n + 1)] = np.eye(n + 1)
f1[n + 1 : 2 * (n + 1)] = 0

L2[n + 1 : 2 * (n + 1), :] = 0  # Streamfunction y = 1, eta = 1
A2[n + 1 : 2 * (n + 1), :] = 0
L2[n + 1 : 2 * (n + 1), n + 1 : 2 * (n + 1)] = np.eye(n + 1)
f2[n + 1 : 2 * (n + 1)] = 0

# Solve the system
L_complete = np.block([[L1, A1], [A2, L2]])
f = np.concatenate([f1, f2])
psi = np.linalg.solve(L_complete, f)

plt.spy(L_complete)
plt.show()

# Convert streamfunction to velocities
u = np.concatenate(
    [2 * eps * Dy @ psi[:block_len], 2 * eps * Dy @ psi[block_len:]]
)
v = np.concatenate(
    [
        -(4 * eps / l) * Dx @ psi[:block_len],
        -(4 * eps / (L - l)) * Dx @ psi[block_len:],
    ]
)

# Remove virtual grid points
u = u[interior]
v = v[interior]
xf_i = xf[interior]
yf_i = yf[interior]

xx_i = xf_i.reshape((2 * (n - 1), n - 1))
yy_i = yf_i.reshape((2 * (n - 1), n - 1))
xx_std_i = np.hstack((xx_i[: n - 1, :], xx_i[n - 1 :, :]))
yy_std_i = np.hstack((yy_i[: n - 1, :], yy_i[n - 1 :, :]))

# Evaluate the Poiseuille flow
pois = (1 - yy_i**2) / 2

# Reshape to get solution grids
uu = u.reshape((2 * (n - 1), n - 1))
vv = v.reshape((2 * (n - 1), n - 1))
psi = psi.reshape((2 * (n + 1), n + 1))
uu_std = np.hstack((uu[: n - 1, :], uu[n - 1 :, :]))
vv_std = np.hstack((vv[: n - 1, :], vv[n - 1 :, :]))
psi_std = np.hstack((psi[: n + 1, :], psi[n + 1 :, :]))

# Change the plotting order of points
xx_std_i = np.fliplr(xx_std_i)
yy_std_i = np.fliplr(yy_std_i)
uu_std = np.fliplr(uu_std)
vv_std = np.fliplr(vv_std)

# Reflect solution to get full domain
xx_full = np.block(
    [
        [xx_std_i, -np.fliplr(xx_std_i)],
        [np.flipud(xx_std_i), -np.fliplr(np.flipud(xx_std_i))],
    ]
)
yy_full = np.block(
    [
        [yy_std_i, np.fliplr(yy_std_i)],
        [-np.flipud(yy_std_i), np.fliplr(-np.flipud(yy_std_i))],
    ]
)

pois_full = (1 - yy_full**2) / 2

uu_full = (
    np.block(
        [
            [uu_std, np.fliplr(uu_std)],
            [np.flipud(uu_std), np.fliplr(np.flipud(uu_std))],
        ]
    )
    + pois_full
)
vv_full = np.block(
    [
        [vv_std, -np.fliplr(vv_std)],
        [-np.flipud(vv_std), np.fliplr(np.flipud(vv_std))],
    ]
)

# Plot the solution
fig, ax = plt.subplots(figsize=(6, 6))
plt.quiver(
    xx_full * H,
    yy_full * H,
    uu_full * H**2,
    vv_full * H**2,
    scale_units="xy",
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Velocity field")
plt.show()

# Plot u surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    xx_full * H,
    yy_full * H,
    uu_full * H**2,
    rstride=1,
    cstride=1,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("Horizontal velocity")

plt.show()

# Plot v surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    xx_full * H,
    yy_full * H,
    vv_full * H**2,
    rstride=1,
    cstride=1,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("Vertical velocity")

plt.show()
