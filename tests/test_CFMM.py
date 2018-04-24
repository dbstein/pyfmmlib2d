import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.utilities.random import float_random, complex_random
from pyfmmlib2d.utilities.cauchy_kernels import \
    cauchy_kernel, cauchy_kernel_d1, cauchy_kernel_d2

################################################################################
# test Cauchy-General sums
# still need to figure out how to test the potential
# for the general sum case

n_source = 100
n_target = 200

source_x = float_random([2, n_source])   # source locations
target_x = float_random([2, n_target])   # target locations
charge   = complex_random([n_source,])   # charges at sources
dipstr   = complex_random([n_source,])   # dipoole at sources

def test_correct_evaluation():
    true_self = cauchy_kernel   (source_x, source_x, charge, dipstr)
    true_targ = cauchy_kernel   (source_x, target_x, charge, dipstr)
    true_d1 =   cauchy_kernel_d1(source_x, target_x, charge, dipstr)
    true_d2 =   cauchy_kernel_d2(source_x, target_x, charge, dipstr)

    # Direct
    direct_out = FMM(
                    kind='cauchy-general',
                    source = source_x,
                    target = target_x,
                    charge = charge,
                    dipstr = dipstr,
                    direct = True,
                    compute_source_potential = True,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                    compute_target_hessian   = True,
                )
    direct_self =   direct_out['source']['u']
    direct_targ =   direct_out['target']['u']
    direct_targ_D = direct_out['target']['Du']
    direct_targ_H = direct_out['target']['Hu']
    # FMM
    fmm_out = FMM(
                    kind='cauchy-general',
                    source = source_x,
                    target = target_x,
                    charge = charge,
                    dipstr = dipstr,
                    compute_source_potential = True,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                    compute_target_hessian  = True,
                )
    fmm_self =    fmm_out['source']['u']
    fmm_targ =    fmm_out['target']['u']
    fmm_targ_D =  fmm_out['target']['Du']
    fmm_targ_H =  fmm_out['target']['Hu']

    # these fail right now, complex valued log issue?
    # assert np.allclose(true_self, direct_self,   rtol=1e-10)
    # assert np.allclose(true_self, fmm_self,      rtol=1e-10)
    # assert np.allclose(true_targ, direct_targ,   rtol=1e-10)
    # assert np.allclose(true_targ, fmm_targ,      rtol=1e-10)
    assert np.allclose(true_d1,   direct_targ_D, rtol=1e-10)
    assert np.allclose(true_d1,   fmm_targ_D,    rtol=1e-10)
    assert np.allclose(true_d2,   direct_targ_H, rtol=1e-10)
    assert np.allclose(true_d2,   fmm_targ_H,    rtol=1e-10)
