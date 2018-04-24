import numpy as np
from pyfmmlib2d import FMM
from pyfmmlib2d.utilities.random import float_random, complex_random
from pyfmmlib2d.utilities.stokes_kernels import \
    stokes_kernel, stokes_kernel_stress

################################################################################
# test Stokes sums

n_source = 100
n_target = 200

source_x = float_random([2, n_source]) # source locations
target_x = float_random([2, n_target]) # target locations
forces   = float_random([2, n_source]) # charges at sources
dipstr   = float_random([2, n_source]) # dipoole strength at sources
dipvec   = float_random([2, n_source]) # dipoole orientation at sources

def test_correct_evaluation():
    true_self =   stokes_kernel(source_x, source_x, forces, dipstr, dipvec)
    true_targ =   stokes_kernel(source_x, target_x, forces, dipstr, dipvec)
    true_stress = stokes_kernel_stress(source_x, target_x, forces, dipstr, dipvec)
    true_self_u = true_self[0]
    true_self_v = true_self[1]
    true_targ_u = true_targ[0]
    true_targ_v = true_targ[1]
    true_ux     = true_stress[0]
    true_uy     = true_stress[1]
    true_vx     = true_stress[2]
    true_vy     = true_stress[3]
    true_p      = true_stress[4]

    # FMM
    fmm_out = FMM(
                    kind='stokes',
                    source = source_x,
                    target = target_x,
                    forces = forces,
                    dipstr = dipstr,
                    dipvec = dipvec,
                    compute_source_velocity = True,
                    compute_target_velocity = True,
                    compute_target_stress   = True,
                )
    fmm_self_u =  fmm_out['source']['u']
    fmm_self_v =  fmm_out['source']['v']
    fmm_targ_u =  fmm_out['target']['u']
    fmm_targ_v =  fmm_out['target']['v']
    fmm_targ_ux = fmm_out['target']['u_x']
    fmm_targ_uy = fmm_out['target']['u_y']
    fmm_targ_vx = fmm_out['target']['v_x']
    fmm_targ_vy = fmm_out['target']['v_y']
    fmm_targ_p =  fmm_out['target']['p']

    # demean the two pressure fields (doesn't actually seem to be necessary)
    true_p -= np.mean(true_p)
    fmm_targ_p -= np.mean(fmm_targ_p)

    assert np.allclose(true_self_u, fmm_self_u,  rtol=1e-10)
    assert np.allclose(true_self_v, fmm_self_v,  rtol=1e-10)
    assert np.allclose(true_targ_u, fmm_targ_u,  rtol=1e-10)
    assert np.allclose(true_targ_v, fmm_targ_v,  rtol=1e-10)
    assert np.allclose(true_ux,     fmm_targ_ux, rtol=1e-10)
    assert np.allclose(true_uy,     fmm_targ_uy, rtol=1e-10)
    assert np.allclose(true_vx,     fmm_targ_vx, rtol=1e-10)
    assert np.allclose(true_vy,     fmm_targ_vy, rtol=1e-10)
    assert np.allclose(true_p,      fmm_targ_p,  rtol=1e-10)
