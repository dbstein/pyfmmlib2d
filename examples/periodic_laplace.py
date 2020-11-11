import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.periodized.real_laplace import periodized_laplace_fmm
from pyfmmlib2d.utilities.random import float_random, complex_random
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

################################################################################
# Periodic Laplace FMM

n_source = 1000
n_grid   = 100

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
# get charges and dipstrs (charge must have mean 0)
charge = np.sin(2*theta+np.pi/3)*theta_h
dipstr = -np.exp(np.cos(2*theta))*theta_h

pfmm = periodized_laplace_fmm([0,np.e,0,np.e], 32)
out = pfmm(source, target=grid, charge=charge, dipstr=dipstr, dipvec=source_normal, compute_target_potential=True)
u = out['target']['u'].reshape([n_grid, n_grid])

# plot the solution, check that it's periodic
fig, ax = plt.subplots()
ax.imshow(u)

# plot the setup
fig, ax = plt.subplots()
ax.scatter(source[0], source[1], color='black')
ax.scatter(pfmm.check[0], pfmm.check[1], color='red')
ax.quiver(pfmm.check[0], pfmm.check[1], pfmm.normals[0], pfmm.normals[1] , color='pink')
ax.scatter(pfmm.source[0], pfmm.source[1], color='blue')
