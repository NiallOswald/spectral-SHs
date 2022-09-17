"""Find the flow of a channel with longitudinal grooves on both walls."""

import numpy as np
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Setting parameters
H = 1
E = 2
e = 1.5

L = E / H
l = e / H  # noqa: E741
delta = e / E

theta = 3 * np.pi / 4
mu = theta - np.pi / 2

# Construct the differentiation matrix
n = 10
D, c = cheb(n)

# Construct the grid of points
x1 = l / 4 * (c - 1)
x2 = ((L - l) * c - l - L) / 4

x = np.concatenate((x1, x2))

alpha = (l * (1 / 4 - (c - 1) ** 2 / 16) * np.tan(mu) + 1) ** -1

y2 = (c + 1) / 2

xx1 = np.tile(x1, (n + 1, 1))
yy1 = (c.reshape(n + 1, 1) + 1) / 2 @ (alpha**-1).reshape(1, n + 1)
xx2, yy2 = np.meshgrid(x2, y2)

xx_std = np.hstack((xx2, xx1))
yy_std = np.hstack((yy2, yy1))

xx = np.vstack((xx1, xx2))
yy = np.vstack((yy1, yy2))

x1f = xx1.flatten()
y1f = yy1.flatten()
x2f = xx2.flatten()
y2f = yy2.flatten()

xf = xx.flatten()
yf = yy.flatten()
