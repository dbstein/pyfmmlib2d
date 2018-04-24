import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.utilities.random import float_random, complex_random
from pyfmmlib2d.utilities.laplace_kernels import \
    laplace_kernel, laplace_kernel_gradient, laplace_kernel_hessian

################################################################################
# test Real-Laplace sums

n_source = 100
n_target = 200

source_x = float_random([2, n_source])   # source locations
target_x = float_random([2, n_target])   # target locations
charge =   float_random([n_source,])     # charge density at sources
dipstr =   float_random([n_source,])     # dipole strength at sources
dipvec =   float_random([2, n_source])   # dipole direction at sources

def test_correct_evaluation():
    true_self = laplace_kernel(source_x, source_x, charge, dipstr, dipvec)
    true_targ = laplace_kernel(source_x, target_x, charge, dipstr, dipvec)
    true_x, true_y = \
        laplace_kernel_gradient(source_x, target_x, charge, dipstr, dipvec)
    true_xx, true_xy, true_yy = \
        laplace_kernel_hessian(source_x, target_x, charge, dipstr, dipvec)

    # Direct
    direct_out = FMM(
                    kind='laplace',
                    source = source_x,
                    target = target_x,
                    charge = charge,
                    dipstr = dipstr,
                    dipvec = dipvec,
                    direct = True,
                    compute_source_potential = True,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                    compute_target_hessian  = True,
                )
    direct_self =    direct_out['source']['u']
    direct_targ =    direct_out['target']['u']
    direct_targ_x =  direct_out['target']['u_x']
    direct_targ_y =  direct_out['target']['u_y']
    direct_targ_xx = direct_out['target']['u_xx']
    direct_targ_xy = direct_out['target']['u_xy']
    direct_targ_yy = direct_out['target']['u_yy']
    # FMM
    fmm_out = FMM(
                    kind='laplace',
                    source = source_x,
                    target = target_x,
                    charge = charge,
                    dipstr = dipstr,
                    dipvec = dipvec,
                    compute_source_potential = True,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                    compute_target_hessian  = True,
                )
    fmm_self =    fmm_out['source']['u']
    fmm_targ =    fmm_out['target']['u']
    fmm_targ_x =  fmm_out['target']['u_x']
    fmm_targ_y =  fmm_out['target']['u_y']
    fmm_targ_xx = fmm_out['target']['u_xx']
    fmm_targ_xy = fmm_out['target']['u_xy']
    fmm_targ_yy = fmm_out['target']['u_yy']

    assert np.allclose(true_self, direct_self,    rtol=1e-10)
    assert np.allclose(true_self, fmm_self,       rtol=1e-10)
    assert np.allclose(true_targ, direct_targ,    rtol=1e-10)
    assert np.allclose(true_targ, fmm_targ,       rtol=1e-10)
    assert np.allclose(true_x,    direct_targ_x,  rtol=1e-10)
    assert np.allclose(true_x,    fmm_targ_x,     rtol=1e-10)
    assert np.allclose(true_y,    direct_targ_y,  rtol=1e-10)
    assert np.allclose(true_y,    fmm_targ_y,     rtol=1e-10)
    assert np.allclose(true_xx,   direct_targ_xx, rtol=1e-10)
    assert np.allclose(true_xx,   fmm_targ_xx,    rtol=1e-10)
    assert np.allclose(true_xy,   direct_targ_xy, rtol=1e-10)
    assert np.allclose(true_xy,   fmm_targ_xy,    rtol=1e-10)
    assert np.allclose(true_yy,   direct_targ_yy, rtol=1e-10)
    assert np.allclose(true_yy,   fmm_targ_yy,    rtol=1e-10)
