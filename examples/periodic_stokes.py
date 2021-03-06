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

dipoles = False # turns graphics for pressure off if using dipoles

# get test grid
grid_v, grid_h = np.linspace(0, np.e, n_grid, endpoint=False, retstep=True)
grid_x, grid_y = np.meshgrid(grid_v, grid_v, indexing='ij')
grid = np.row_stack([grid_x.ravel(), grid_y.ravel()])

# get sources on circle of radius 0.7
theta, theta_h = np.linspace(0, 2*np.pi, n_source, endpoint=False, retstep=True)
source_x = np.e/2 + grid_h*0 + 0.9*np.cos(theta)
source_y = np.e/2 + grid_h*0 + 0.9*np.sin(theta)
source = np.row_stack([source_x, source_y])
# get normals
source_x_normal = np.cos(theta)
source_y_normal = np.sin(theta)
source_normal = np.row_stack([source_x_normal, source_y_normal])
# get forces and dipstrs
forces = np.row_stack([ np.exp(np.sin(2*theta+np.pi/3))-1.0, 0.1 + np.cos(2*theta+np.pi/3) ])*theta_h
dipstr = np.row_stack([ np.exp(np.cos(2*theta)), np.cos(2*theta+np.pi/3) ])*theta_h*int(dipoles)

pfmm = periodized_stokes_fmm([0,np.e,0,np.e])
out = pfmm(source+np.e, forces=forces, dipstr=dipstr, dipvec=source_normal, compute_target_velocity=True, compute_target_stress=True, target=grid+2*np.e)
forcetot = np.sum(forces, axis=1)

u = out['target']['u'].reshape([n_grid, n_grid])
v = out['target']['v'].reshape([n_grid, n_grid])
p = out['target']['p'].reshape([n_grid, n_grid])

fig, ax = plt.subplots()
if not dipoles:
	ax.pcolormesh(grid_x, grid_y, p)
ax.quiver(grid_x, grid_y, u, v, color='black' if dipoles else 'white')
