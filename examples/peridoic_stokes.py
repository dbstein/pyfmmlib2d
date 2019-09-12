import numpy as np
from pyfmmlib2d import SFMM
from pyfmmlib2d.periodized.stokes import periodized_stokes_fmm
from pyfmmlib2d.utilities.random import float_random, complex_random
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

################################################################################
# Periodic Laplace FMM

n_source = 1000
n_grid   = 50

# get test grid
grid_v = np.linspace(0, np.e, n_grid, endpoint=False)
grid_x, grid_y = np.meshgrid(grid_v, grid_v, indexing='ij')
grid = np.row_stack([grid_x.ravel(), grid_y.ravel()])

# get sources on circle of radius 0.7
theta, theta_h = np.linspace(0, 2*np.pi, n_source, endpoint=False, retstep=True)
source_x = 3*np.e/4 + 0.9*np.cos(theta)
source_y = 1.2 + 0.9*np.sin(theta)
source = np.row_stack([source_x, source_y])
# get normals
source_x_normal = np.cos(theta)
source_y_normal = np.sin(theta)
source_normal = np.row_stack([source_x_normal, source_y_normal])
# get forces and dipstrs
forces = np.row_stack([ -np.exp(np.sin(2*theta+np.pi/3)), np.cos(2*theta+np.pi/3) ])*theta_h
dipstr = np.row_stack([ np.exp(np.cos(2*theta)), np.cos(2*theta+np.pi/3) ])*theta_h

pfmm = periodized_stokes_fmm([0,np.e,0,np.e], 64, 1e-14)
out = pfmm(source, grid, forces=forces, dipstr=dipstr, dipvec=source_normal)

u = out['target']['u'].reshape([n_grid, n_grid])
v = out['target']['v'].reshape([n_grid, n_grid])
p = out['target']['p'].reshape([n_grid, n_grid])

# plot the solution, check that it's periodic
fig, ax = plt.subplots()
ax.pcolormesh(grid_x, grid_y, p, vmin=-5, vmax=5)
ax.quiver(grid_x, grid_y, u, v, color='white')
