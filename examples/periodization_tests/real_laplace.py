import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.periodized.real_laplace import periodized_laplace_fmm
from pyfmmlib2d.utilities.random import float_random, complex_random
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfmmlib2d import RFMM
import time
plt.ion()

################################################################################
# Periodic Laplace FMM

# tested against FFT based grid solve; compared outside of support of forces
# the grid has been spaced so a grid node comes within ~10^{-12} of a check pt
# should work in all corners and sides

n_grid = 150

# get uniform grid on [0, 2π]
grid_v, grid_h = np.linspace(0, 2*np.pi, n_grid, endpoint=False, retstep=True)
grid_xv = grid_v + 3.381898123243e-02
grid_yv = grid_v + 1e-12
grid_x, grid_y = np.meshgrid(grid_xv, grid_yv, indexing='ij')
grid = np.row_stack([grid_x.ravel(), grid_y.ravel()])
k = np.fft.fftfreq(n_grid, grid_h/(2*np.pi))
kx, ky = np.meshgrid(k, k, indexing='ij')
k2 = kx*kx + ky*ky
k2[0,0] = np.Inf
ilap = -1.0/k2

# put a test pulse at location in center
k = 30

for center_x in (0.0, np.pi, 2*np.pi):
	for center_y in (0.0, np.pi, 2*np.pi):

		print('\nCenter x: {:0.2f}'.format(center_x))
		print(  'Center y: {:0.2f}'.format(center_y))

		def get_shift(xs, ys):
			lax = center_x+0.2 + xs
			lay = center_y+0.2 + ys
			lbx = center_x-0.2 + xs
			lby = center_y-0.2 + ys
			d2a = (grid_x-lax)**2 + (grid_y-lay)**2
			d2b = (grid_x-lbx)**2 + (grid_y-lby)**2

			fa = np.exp(-k*d2a)
			fb = -np.exp(-k*d2b)
			return fa + fb

		f = np.zeros_like(grid_x)
		shift_vec = [-2*np.pi, 0.0, 2*np.pi]
		for xs in shift_vec:
			for ys in shift_vec:
				f += get_shift(xs, ys)

		# solve Poisson problem on grid
		ua = np.fft.ifft2(ilap*np.fft.fft2(f)).real

		# get charge
		qw = grid_h**2
		charge = f.ravel() * qw / (2*np.pi)

		# evaluate FMM
		st1 = time.time()
		pfmm = periodized_laplace_fmm(p=16, N=4)
		st2 = time.time()
		out = pfmm(grid, charge=charge, compute_source_potential=True)
		periodized_time_with_setup = time.time() - st1
		periodized_time_without_setup = time.time() - st2
		ue = out['self']['u'].reshape([n_grid, n_grid])

		# compare to raw FMM speed
		st = time.time()
		_ = RFMM(grid, grid, charge=charge, compute_source_potential=True)
		raw_time = time.time() - st

		# now ignore everything well outside of the support of f
		r = 2
		bad = np.zeros(grid_x.shape, dtype=bool)
		for xs in shift_vec:
			for ys in shift_vec:
				d2 = (grid_x-center_x-xs)**2 + (grid_y-center_y-ys)**2
				bad = np.logical_or(bad, d2<r)
		good = ~bad

		# get error with adjusting mean
		ue -= np.mean(ue)
		error = np.abs(ue-ua)/np.abs(ua[good]).max()
		me = np.ma.array(error, mask=bad)
		emax = me.max()

		if center_x == 0 and center_y == 0:

			# plot error
			fig, ax = plt.subplots()
			clf = ax.pcolormesh(grid_x, grid_y, me, norm=mpl.colors.LogNorm())
			plt.colorbar(clf)

		print('    Error = {:0.2e}'.format(emax))
		print('    Time, with setup: {:0.2f}'.format(periodized_time_with_setup))
		print('    Time, no setup:   {:0.2f}'.format(periodized_time_without_setup))
		print('    Time, freespace:  {:0.2f}'.format(raw_time))

