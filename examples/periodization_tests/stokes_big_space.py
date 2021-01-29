import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.periodized.stokes import periodized_stokes_fmm
from pyfmmlib2d.utilities.random import float_random, complex_random
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfmmlib2d import SFMM
import time
plt.ion()

################################################################################
# Periodic Stokes FMM

# tested against FFT based grid solve; compared outside of support of forces
# the grid has been spaced so a grid node comes within ~10^{-12} of a check pt
# should work in all corners and sides

n_grid = 300

vmin = -1
vmax = 11
vran = vmax - vmin
vmid = vmin + 0.5*vran

# get uniform grid on [vmin, vmax]
grid_v, grid_h = np.linspace(vmin, vmax, n_grid, endpoint=False, retstep=True)
grid_xv = grid_v + 3.381898123243e-02
grid_yv = grid_v + 1e-12
grid_x, grid_y = np.meshgrid(grid_xv, grid_yv, indexing='ij')
grid = np.row_stack([grid_x.ravel(), grid_y.ravel()])
k = np.fft.fftfreq(n_grid, grid_h/(2*np.pi))
kx, ky = np.meshgrid(k, k, indexing='ij')
k2 = kx*kx + ky*ky
k2[0,0] = np.Inf
ilap = -1.0/k2
ikx = 1j*kx
iky = 1j*ky

# put a test pulse at location in center
k = 30

for center_x in (vmin, vmid, vmax):
	for center_y in (vmin, vmid, vmax):

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
		shift_vec = [-vran, 0.0, vran]
		for xs in shift_vec:
			for ys in shift_vec:
				f += get_shift(xs, ys)
		fx = f.copy()
		fy = f.copy()

		# solve Stokes problem on grid
		fxh = np.fft.fft2(fx)
		fyh = np.fft.fft2(fy)
		div_fh = ikx*fxh + iky*fyh
		ph = ilap*div_fh
		uh = ilap*(ikx*ph - fxh)
		vh = ilap*(iky*ph - fyh)
		ua = np.fft.ifft2(uh).real
		va = np.fft.ifft2(vh).real
		pa = np.fft.ifft2(ph).real

		# get weighted forces
		qw = grid_h**2
		forcex = fx.ravel() * qw
		forcey = fy.ravel() * qw
		forces = np.row_stack([forcex, forcey])

		# evaluate FMM
		st1 = time.time()
		pfmm = periodized_stokes_fmm([vmin, vmax, vmin, vmax], p=16, N=4)
		st2 = time.time()
		out = pfmm(grid, forces=forces, compute_source_velocity=True, compute_source_stress=True)
		periodized_time_with_setup = time.time() - st1
		periodized_time_without_setup = time.time() - st2
		ue = out['self']['u'].reshape([n_grid, n_grid])
		ve = out['self']['v'].reshape([n_grid, n_grid])
		pe = out['self']['p'].reshape([n_grid, n_grid])

		# now ignore everything well outside of the support of f
		r = 2
		bad = np.zeros(grid_x.shape, dtype=bool)
		for xs in shift_vec:
			for ys in shift_vec:
				d2 = (grid_x-center_x-xs)**2 + (grid_y-center_y-ys)**2
				bad = np.logical_or(bad, d2<r)
		good = ~bad

		# get error without adjusting mean!  should already be fixed to 0
		uerror = np.abs(ue-ua)/np.abs(ua[good]).max()
		mue = np.ma.array(uerror, mask=bad)
		uemax = mue.max()
		verror = np.abs(ve-va)/np.abs(va[good]).max()
		mve = np.ma.array(verror, mask=bad)
		vemax = mve.max()
		pe -= np.mean(pe)
		perror = np.abs(pe-pa)/np.abs(pa[good]).max()
		mpe = np.ma.array(perror, mask=bad)
		pemax = mpe.max()

		# compare to raw FMM speed
		st = time.time()
		_ = SFMM(grid, grid, forces=forces, compute_source_velocity=True, compute_source_stress=True)
		raw_time = time.time() - st

		if center_x == 0 and center_y == 0:

			# plot error
			fig, ax = plt.subplots()
			clf = ax.pcolormesh(grid_x, grid_y, mue, norm=mpl.colors.LogNorm())
			ax.set_title('Error, $u$')
			plt.colorbar(clf)

			fig, ax = plt.subplots()
			clf = ax.pcolormesh(grid_x, grid_y, mve, norm=mpl.colors.LogNorm())
			ax.set_title('Error, $v$')
			plt.colorbar(clf)

			fig, ax = plt.subplots()
			clf = ax.pcolormesh(grid_x, grid_y, mpe, norm=mpl.colors.LogNorm())
			ax.set_title('Error, $p$')
			plt.colorbar(clf)

		print('    Error, u = {:0.2e}'.format(uemax))
		print('    Error, v = {:0.2e}'.format(vemax))
		print('    Error, p = {:0.2e}'.format(pemax))
		print('    Time, with setup: {:0.2f}'.format(periodized_time_with_setup))
		print('    Time, no setup:   {:0.2f}'.format(periodized_time_without_setup))
		print('    Time, freespace:  {:0.2f}'.format(raw_time))

