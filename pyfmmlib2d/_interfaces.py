import numpy as np

from fmmlib2d import hfmm2dparttarg
from fmmlib2d import lfmm2dparttarg
from fmmlib2d import rfmm2dparttarg
from fmmlib2d import zfmm2dparttarg
from fmmlib2d import cfmm2dparttarg

from fmmlib2d import h2dpartdirect
from fmmlib2d import l2dpartdirect
from fmmlib2d import r2dpartdirect
from fmmlib2d import z2dpartdirect
from fmmlib2d import c2dpartdirect

from stokesfmm import bhfmm2dparttarg

def initialize_precision(iprec):
    if iprec < -2:
        iprec = -2
    if iprec > 5:
        iprec = 5
    return int(iprec)
def get_dummy_shape(shape):
    if len(shape) == 1:
        return (1,)
    else:
        return (shape[0], 1)
def check_array(arr, shape, mytype, name, return_2dim=False):
    """
    Function to make sure an array is acceptable
    If arr is None:
        returns dummy array, int(0)
    Else:
        Throws error if:
            not a numpy array
            shape is not correct
                note: if shape is (k,None), won't check one second dim size
            type is not as specified
                will cast float to complex without an error
                will NOT cast complex to float
        Reallocates if:
            not Fortran contiguous
        Note: name is just to help throw useful errors to the user
        Returns:
            original array or reallocated array, int(1)
    If return_2dim, also returns the second dim size
    """
    if arr is None:
        sh = get_dummy_shape(shape)
        arr = np.empty(sh, order='F', dtype=mytype)
        here = int(0)
    else:
        if type(arr) != np.ndarray:
            raise Exception(name + ' must be numpy array')
        sh = arr.shape
        if len(sh) != len(shape) or sh[0] != shape[0]:
            raise Exception(name + 'does not have correct shape.')
        if len(sh) > 1 and shape[1] != None and sh[1] != shape[1]:
            raise Exception(name + 'does not have correct shape.')
        if arr.dtype != mytype:
            if not (arr.dtype == float and mytype == complex):
                raise Exception(name + ' must have type ' + str(mytype))
        arr = arr.astype(mytype, order='F', copy=False)
        here = int(1)
    if return_2dim:
        return arr, here, sh[1]
    else:
        return arr, here
def check_output(arr, used, shape, mytype):
    """
    Function to check on output array
    If used is False:
        returns dummy array, int(0)
    If used is True:
        If arr:
            has the right shape
            has the right type
            is Fortran contiguous
            returns arr, int(1)
        Else:
            allocates a new array with the right shape, type, contiguity
            returns the new array, int(1)
    """
    if used:
        if arr is not None:
            test1 = arr.shape == shape
            test2 = arr.dtype == mytype
            test3 = arr.flags['F_CONTIGUOUS']
            if not (test1 and test2 and test3):
                arr = None
        if arr is None:
            arr = np.empty(shape, order='F', dtype=mytype)
        else:
            arr = arr.astype(mytype, order='F', copy=False)
    else:
        arr = np.empty(get_dummy_shape(shape), order='F', dtype=mytype)
    return arr, int(used)
def get_fmmlib2d_output(csp,csg,csh,sp,sg,sh,ctp,ctg,cth,tp,tg,th,ier):
    output = {}
    any_source = csp or csg or csh
    if any_source:
        source_output = {}
        if csp:
            source_output['Pu'] = sp
            source_output['u']  = sp[0]
        if csg:
            source_output['Du']  = sg
            source_output['u_x'] = sg[0]
            source_output['u_y'] = sg[1]
        if csh:
            source_output['Hu']   = sh
            source_output['u_xx'] = sh[0]
            source_output['u_xy'] = sh[1]
            source_output['u_yx'] = sh[1]
            source_output['u_yy'] = sh[2]
        output['source'] = source_output
    any_target = ctp or ctg or cth
    if any_target:
        target_output = {}
        if ctp:
            target_output['Pu'] = tp
            target_output['u']  = tp[0]
        if ctg:
            target_output['Du']  = tg
            target_output['u_x'] = tg[0]
            target_output['u_y'] = tg[1]
        if cth:
            target_output['Hu']   = th
            target_output['u_xx'] = th[0]
            target_output['u_xy'] = th[1]
            target_output['u_yx'] = th[1]
            target_output['u_yy'] = th[2]
        output['target'] = target_output
    output['ier'] = ier
    return output
def get_fmmlib2d_output_cauchy(csp,csg,csh,sp,sg,sh,ctp,ctg,cth,tp,tg,th,ier):
    output = {}
    any_source = csp or csg or csh
    if any_source:
        source_output = {}
        if csp:
            source_output['u']  = sp
        if csg:
            source_output['Du']  = sg
        if csh:
            source_output['Hu']   = sh
        output['source'] = source_output
    any_target = ctp or ctg or cth
    if any_target:
        target_output = {}
        if ctp:
            target_output['u'] = tp
        if ctg:
            target_output['Du']  = tg
        if cth:
            target_output['Hu']   = th
        output['target'] = target_output
    output['ier'] = ier
    return output

def FMM(kind, **kwargs):
    """
    Pythonic interface to Particle FMM Routines for:
        Helmholtz, Laplace, Cauchy, Biharmonic, and Stokes
        This function calls the following functions:
            HFMM: (kind='helmholtz')
            LFMM: (kind='laplace-complex')
            RFMM: (kind='laplace-real' or 'laplace')
            ZFMM: (kind='cauchy')
            CFMM: (kind='cauchy-general')
            BFMM: (kind='biharmonic')
            SFMM: (kind='stokes')
        Please see help files for individual functions for more details.

    Details on the 'precision' parameter (this is the same across all functions):
        precision (optional), int: precision requested of FMM
        -2: least squares errors < 0.5e+00
        -1: least squares errors < 0.5e-01
         0: least squares errors < 0.5e-02
         1: least squares errors < 0.5e-03
         2: least squares errors < 0.5e-06
         3: least squares errors < 0.5e-09
         4: least squares errors < 0.5e-12
         5: least squares errors < 0.5e-15
         note that this is ignored if direct=True
         and defaults to 4 for all routines
    """
    return function_map[kind](**kwargs)

def HFMM(
        source,
        target = None,
        charge = None,
        dipstr = None,
        dipvec = None,
        direct = False,
        compute_source_potential = False,
        compute_source_gradient  = False,
        compute_source_hessian   = False,
        compute_target_potential = False,
        compute_target_gradient  = False,
        compute_target_hessian   = False,
        array_source_potential   = None,
        array_source_gradient    = None,
        array_source_hessian     = None,
        array_target_potential   = None,
        array_target_gradient    = None,
        array_target_hessian     = None,
        precision                = 4,
        helmholtz_parameter      = 1.0,
    ):
    """
    Pythonic interface for Helmholtz Particle FMM
    Wraps the two functions:
        hfmm2dparttarg - (if direct=False)
        h2dpartdirect  - (if direct=True)
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    charge      (optional), complex(ns):  charges at source locations
    dipstr      (optional), complex(ns):  dipole at source locations
    dipvec      (optional), float(2, ns): orientation vector of dipoles
        if dipstr is set, then dipvec must be, also
    direct      (optional), bool:         do direct sum or FMM
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), complex(k,n): preallocated arrays for result
        k = 1 for *=potential, 2 for *=gradient, 3 for *=hessian
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM
    helmholtz_parameter (optional), complex: complex helmholtz parameter

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'Pu'   : complex(1,ns), potential
            'u'    : complex(ns),   potential
            'u_x'  : complex(ns),   x-derivative of potential
            'u_y'  : complex(ns),   y-derivative of potential
            'Du'   : complex(2,ns), gradient of potential
            'u_xx' : complex(ns),   xx-derivative of potential
            'u_xy' : complex(ns),   xy-derivative of potential
            'u_yx' : complex(ns),   yx-derivative of potential
            'u_yy' : complex(ns),   xy-derivative of potential
            'Hu'   : complex(3,ns), hessian of potential
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) 'u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yx', and 'u_yy' are 
                not duplications; they are simply views
                into the arrays 'Pu', Du' and 'Hu', organized as follows:
                Pu[0] = u
                Du[0] = u_x
                Du[1] = u_y
                Hu[0] = u_xx
                Hu[1] = u_xy
                Hu[1] = u_yx
                Hu[2] = u_yy
            2) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['**']"
                will return True (note ** is Pu for *=potential,
                    Du for *=gradient, Hu for *=Hessian)
                If the array was provided but incorrect, then the code
                will return False
            3) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    charge, ifcharge = check_array(charge, (ns,), complex, 'charge')
    dipstr, ifdipstr = check_array(dipstr, (ns,), complex, 'dipstr')
    dipvec, ifdipvec = check_array(dipvec, (2,ns), float, 'dipvec')
    if ifdipstr and not ifdipvec:
        raise Exception('If dipstr is provided, dipvec must be also')
    pot, ifpot = check_output(array_source_potential,
                            compute_source_potential, (1,ns), complex)
    grad, ifgrad = check_output(array_source_gradient,
                            compute_source_gradient, (2,ns), complex)
    hess, ifhess = check_output(array_source_hessian,
                            compute_source_hessian, (3,ns), complex)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_potential or compute_target_gradient \
             or compute_target_hessian:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    pottarg, ifpottarg = check_output(array_target_potential,
                            compute_target_potential, (1,nt), complex)
    gradtarg, ifgradtarg = check_output(array_target_gradient,
                            compute_target_gradient, (2,nt), complex)
    hesstarg, ifhesstarg = check_output(array_target_hessian,
                            compute_target_hessian, (3,nt), complex)
    ier = int(0)
    iprec = initialize_precision(precision)
    zk = complex(helmholtz_parameter)

    if direct:
        h2dpartdirect(zk, ns, source, ifcharge, charge, ifdipstr, dipstr,
            dipvec, ifpot, pot, ifgrad, grad, ifhess, hess, nt, target,
            ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)
    else:
        hfmm2dparttarg(ier, iprec, zk, ns, source, ifcharge, charge,
            ifdipstr, dipstr, dipvec, ifpot, pot, ifgrad, grad, ifhess,
            hess, nt, target, ifpottarg, pottarg, ifgradtarg, gradtarg,
            ifhesstarg, hesstarg)

    out = get_fmmlib2d_output(
            compute_source_potential,
            compute_source_gradient,
            compute_source_hessian,
            pot, grad, hess,
            compute_target_potential,
            compute_target_gradient,
            compute_target_hessian,
            pottarg, gradtarg, hesstarg,
            ier
        )
    return out

def LFMM(
        source,
        target = None,
        charge = None,
        dipstr = None,
        dipvec = None,
        direct = False,
        compute_source_potential = False,
        compute_source_gradient  = False,
        compute_source_hessian   = False,
        compute_target_potential = False,
        compute_target_gradient  = False,
        compute_target_hessian   = False,
        array_source_potential   = None,
        array_source_gradient    = None,
        array_source_hessian     = None,
        array_target_potential   = None,
        array_target_gradient    = None,
        array_target_hessian     = None,
        precision                = 4,
    ):
    """
    Pythonic interface for Laplace Particle FMM (complex densities)
    Wraps the two functions:
        lfmm2dparttarg - (if direct=False)
        l2dpartdirect  - (if direct=True)
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    charge      (optional), complex(ns):  charges at source locations
    dipstr      (optional), complex(ns):  dipole at source locations
    dipvec      (optional), float(2, ns): orientation vector of dipoles
        if dipstr is set, then dipvec must be, also
    direct      (optional), bool:         do direct sum or FMM
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), complex(k,n): preallocated arrays for result
        k = 1 for *=potential, 2 for *=gradient, 3 for *=hessian
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'Pu'   : complex(1,ns), potential
            'u'    : complex(ns),   potential
            'u_x'  : complex(ns),   x-derivative of potential
            'u_y'  : complex(ns),   y-derivative of potential
            'Du'   : complex(2,ns), gradient of potential
            'u_xx' : complex(ns),   xx-derivative of potential
            'u_xy' : complex(ns),   xy-derivative of potential
            'u_yx' : complex(ns),   yx-derivative of potential
            'u_yy' : complex(ns),   xy-derivative of potential
            'Hu'   : complex(3,ns), hessian of potential
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) 'u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yx', and 'u_yy' are 
                not duplications; they are simply views
                into the arrays 'Pu', Du' and 'Hu', organized as follows:
                Pu[0] = u
                Du[0] = u_x
                Du[1] = u_y
                Hu[0] = u_xx
                Hu[1] = u_xy
                Hu[1] = u_yx
                Hu[2] = u_yy
            2) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['**']"
                will return True (note ** is Pu for *=potential,
                    Du for *=gradient, Hu for *=Hessian)
                If the array was provided but incorrect, then the code
                will return False
            3) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    charge, ifcharge = check_array(charge, (ns,), complex, 'charge')
    dipstr, ifdipstr = check_array(dipstr, (ns,), complex, 'dipstr')
    dipvec, ifdipvec = check_array(dipvec, (2,ns), float, 'dipvec')
    if ifdipstr and not ifdipvec:
        raise Exception('If dipstr is provided, dipvec must be also')
    pot, ifpot = check_output(array_source_potential,
                            compute_source_potential, (1,ns), complex)
    grad, ifgrad = check_output(array_source_gradient,
                            compute_source_gradient, (2,ns), complex)
    hess, ifhess = check_output(array_source_hessian,
                            compute_source_hessian, (3,ns), complex)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_potential or compute_target_gradient \
             or compute_target_hessian:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    pottarg, ifpottarg = check_output(array_target_potential,
                            compute_target_potential, (1,nt), complex)
    gradtarg, ifgradtarg = check_output(array_target_gradient,
                            compute_target_gradient, (2,nt), complex)
    hesstarg, ifhesstarg = check_output(array_target_hessian,
                            compute_target_hessian, (3,nt), complex)
    ier = int(0)
    iprec = initialize_precision(precision)

    if direct:
        l2dpartdirect(ns, source, ifcharge, charge, ifdipstr, dipstr, dipvec,
            ifpot, pot, ifgrad, grad, ifhess, hess, nt, target, ifpottarg,
            pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)
    else:
        lfmm2dparttarg(ier, iprec, ns, source, ifcharge, charge, ifdipstr,
            dipstr, dipvec, ifpot, pot, ifgrad, grad, ifhess, hess, nt, target,
            ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)

    out = get_fmmlib2d_output(
            compute_source_potential,
            compute_source_gradient,
            compute_source_hessian,
            pot, grad, hess,
            compute_target_potential,
            compute_target_gradient,
            compute_target_hessian,
            pottarg, gradtarg, hesstarg,
            ier
        )
    return out

def RFMM(
        source,
        target = None,
        charge = None,
        dipstr = None,
        dipvec = None,
        direct = False,
        compute_source_potential = False,
        compute_source_gradient  = False,
        compute_source_hessian   = False,
        compute_target_potential = False,
        compute_target_gradient  = False,
        compute_target_hessian   = False,
        array_source_potential   = None,
        array_source_gradient    = None,
        array_source_hessian     = None,
        array_target_potential   = None,
        array_target_gradient    = None,
        array_target_hessian     = None,
        precision                = 4,
    ):
    """
    Pythonic interface for Laplace Particle FMM (real densities)
    Wraps the two functions:
        rfmm2dparttarg - (if direct=False)
        r2dpartdirect  - (if direct=True)
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    charge      (optional), float(ns):    charges at source locations
    dipstr      (optional), float(ns):    dipole at source locations
    dipvec      (optional), float(2, ns): orientation vector of dipoles
        if dipstr is set, then dipvec must be, also
    direct      (optional), bool:         do direct sum or FMM
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), float(k,n):   preallocated arrays for result
        k = 1 for *=potential, 2 for *=gradient, 3 for *=hessian
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'Pu'   : complex(1,ns), potential
            'u'    : complex(ns),   potential
            'u_x'  : float(ns),   x-derivative of potential
            'u_y'  : float(ns),   y-derivative of potential
            'Du'   : float(2,ns), gradient of potential
            'u_xx' : float(ns),   xx-derivative of potential
            'u_xy' : float(ns),   xy-derivative of potential
            'u_yx' : float(ns),   yx-derivative of potential
            'u_yy' : float(ns),   xy-derivative of potential
            'Hu'   : float(3,ns), hessian of potential
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) 'u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yx', and 'u_yy' are 
                not duplications; they are simply views
                into the arrays 'Pu', Du' and 'Hu', organized as follows:
                Pu[0] = u
                Du[0] = u_x
                Du[1] = u_y
                Hu[0] = u_xx
                Hu[1] = u_xy
                Hu[1] = u_yx
                Hu[2] = u_yy
            2) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['**']"
                will return True (note ** is Pu for *=potential,
                    Du for *=gradient, Hu for *=Hessian)
                If the array was provided but incorrect, then the code
                will return False
            3) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    charge, ifcharge = check_array(charge, (ns,), float, 'charge')
    dipstr, ifdipstr = check_array(dipstr, (ns,), float, 'dipstr')
    dipvec, ifdipvec = check_array(dipvec, (2,ns), float, 'dipvec')
    if ifdipstr and not ifdipvec:
        raise Exception('If dipstr is provided, dipvec must be also')
    pot, ifpot = check_output(array_source_potential,
                            compute_source_potential, (1,ns), float)
    grad, ifgrad = check_output(array_source_gradient,
                            compute_source_gradient, (2,ns), float)
    hess, ifhess = check_output(array_source_hessian,
                            compute_source_hessian, (3,ns), float)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_potential or compute_target_gradient \
             or compute_target_hessian:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    pottarg, ifpottarg = check_output(array_target_potential,
                            compute_target_potential, (1,nt), float)
    gradtarg, ifgradtarg = check_output(array_target_gradient,
                            compute_target_gradient, (2,nt), float)
    hesstarg, ifhesstarg = check_output(array_target_hessian,
                            compute_target_hessian, (3,nt), float)
    ier = int(0)
    iprec = initialize_precision(precision)

    if direct:
        r2dpartdirect(ns, source, ifcharge, charge, ifdipstr, dipstr, dipvec,
            ifpot, pot, ifgrad, grad, ifhess, hess, nt, target, ifpottarg,
            pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)
    else:
        rfmm2dparttarg(ier, iprec, ns, source, ifcharge, charge, ifdipstr,
            dipstr, dipvec, ifpot, pot, ifgrad, grad, ifhess, hess, nt, target,
            ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)

    out = get_fmmlib2d_output(
            compute_source_potential,
            compute_source_gradient,
            compute_source_hessian,
            pot, grad, hess,
            compute_target_potential,
            compute_target_gradient,
            compute_target_hessian,
            pottarg, gradtarg, hesstarg,
            ier
        )
    return out

def ZFMM(
        source,
        target = None,
        dipstr = None,
        direct = False,
        compute_source_potential = False,
        compute_source_gradient  = False,
        compute_source_hessian   = False,
        compute_target_potential = False,
        compute_target_gradient  = False,
        compute_target_hessian   = False,
        array_source_potential   = None,
        array_source_gradient    = None,
        array_source_hessian     = None,
        array_target_potential   = None,
        array_target_gradient    = None,
        array_target_hessian     = None,
        precision                = 4,
    ):
    """
    Pythonic interface for Cauchy Particle FMM
    Wraps the two functions:
        zfmm2dparttarg - (if direct=False)
        z2dpartdirect  - (if direct=True)
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    dipstr      (required), complex(ns):  dipole at source locations
    direct      (optional), bool:         do direct sum or FMM
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), complex(nt):  preallocated arrays for result
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'u'    : complex(ns),   potential
            'Du'   : complex(ns),   gradient of potential
            'Hu'   : complex(ns),   hessian of potential
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['**']"
                will return True (note ** is u for *=potential,
                    Du for *=gradient, Hu for *=Hessian)
                If the array was provided but incorrect, then the code
                will return False
            2) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    dipstr, ifdipstr = check_array(dipstr, (ns,), complex, 'dipstr')
    if not ifdipstr:
        raise Exception("For fmm kind 'cauchy', dipstr must be provided")
    pot, ifpot = check_output(array_source_potential,
                            compute_source_potential, (ns,), complex)
    grad, ifgrad = check_output(array_source_gradient,
                            compute_source_gradient, (ns,), complex)
    hess, ifhess = check_output(array_source_hessian,
                            compute_source_hessian, (ns,), complex)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_potential or compute_target_gradient \
             or compute_target_hessian:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    pottarg, ifpottarg = check_output(array_target_potential,
                            compute_target_potential, (nt,), complex)
    gradtarg, ifgradtarg = check_output(array_target_gradient,
                            compute_target_gradient, (nt,), complex)
    hesstarg, ifhesstarg = check_output(array_target_hessian,
                            compute_target_hessian, (nt,), complex)
    ier = int(0)
    iprec = initialize_precision(precision)

    if direct:
        z2dpartdirect(ns, source, dipstr, ifpot, pot, ifgrad, grad, ifhess,
            hess, nt, target, ifpottarg, pottarg, ifgradtarg, gradtarg,
            ifhesstarg, hesstarg)
    else:
        zfmm2dparttarg(ier, iprec, ns, source, dipstr, ifpot, pot, ifgrad, grad,
            ifhess, hess, nt, target, ifpottarg, pottarg, ifgradtarg, gradtarg,
            ifhesstarg, hesstarg)

    out = get_fmmlib2d_output_cauchy(
            compute_source_potential,
            compute_source_gradient,
            compute_source_hessian,
            pot, grad, hess,
            compute_target_potential,
            compute_target_gradient,
            compute_target_hessian,
            pottarg, gradtarg, hesstarg,
            ier
        )
    return out

def CFMM(
        source,
        target = None,
        charge = None,
        dipstr = None,
        direct = False,
        compute_source_potential = False,
        compute_source_gradient  = False,
        compute_source_hessian   = False,
        compute_target_potential = False,
        compute_target_gradient  = False,
        compute_target_hessian   = False,
        array_source_potential   = None,
        array_source_gradient    = None,
        array_source_hessian     = None,
        array_target_potential   = None,
        array_target_gradient    = None,
        array_target_hessian     = None,
        precision                = 4,
    ):
    """
    Pythonic interface for Cauchy Particle FMM (general)
    Wraps the two functions:
        cfmm2dparttarg - (if direct=False)
        c2dpartdirect  - (if direct=True)
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    charge      (optional), complex(ns):  charges at source locations
    dipstr      (optional), complex(ns):  dipole at source locations
    direct      (optional), bool:         do direct sum or FMM
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), complex(k,n): preallocated arrays for result
        k = 1 for *=potential, 2 for *=gradient, 3 for *=hessian
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'u'    : complex(ns),   potential
            'Du'   : complex(ns),   gradient of potential
            'Hu'   : complex(ns),   hessian of potential
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['**']"
                will return True (note ** is u for *=potential,
                    Du for *=gradient, Hu for *=Hessian)
                If the array was provided but incorrect, then the code
                will return False
            2) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    charge, ifcharge = check_array(charge, (ns,), complex, 'charge')
    dipstr, ifdipstr = check_array(dipstr, (ns,), complex, 'dipstr')
    pot, ifpot = check_output(array_source_potential,
                            compute_source_potential, (ns,), complex)
    grad, ifgrad = check_output(array_source_gradient,
                            compute_source_gradient, (ns,), complex)
    hess, ifhess = check_output(array_source_hessian,
                            compute_source_hessian, (ns,), complex)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_potential or compute_target_gradient \
             or compute_target_hessian:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    pottarg, ifpottarg = check_output(array_target_potential,
                            compute_target_potential, (nt,), complex)
    gradtarg, ifgradtarg = check_output(array_target_gradient,
                            compute_target_gradient, (nt,), complex)
    hesstarg, ifhesstarg = check_output(array_target_hessian,
                            compute_target_hessian, (nt,), complex)
    ier = int(0)
    iprec = initialize_precision(precision)

    if direct:
        c2dpartdirect(ns, source, ifcharge, charge, ifdipstr, dipstr, ifpot,
            pot, ifgrad, grad, ifhess, hess, nt, target, ifpottarg, pottarg,
            ifgradtarg, gradtarg, ifhesstarg, hesstarg)
    else:
        cfmm2dparttarg(ier, iprec, ns, source, ifcharge, charge, ifdipstr,
            dipstr, ifpot, pot, ifgrad, grad, ifhess, hess, nt, target,
            ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg)

    out = get_fmmlib2d_output_cauchy(
            compute_source_potential,
            compute_source_gradient,
            compute_source_hessian,
            pot, grad, hess,
            compute_target_potential,
            compute_target_gradient,
            compute_target_hessian,
            pottarg, gradtarg, hesstarg,
            ier
        )
    return out

def BFMM(
        source,
        target =  None,
        charge =  None,
        dipole1 = None,
        dipole2 = None,
        compute_source_velocity =               False,
        compute_source_analytic_gradient =      False,
        compute_source_anti_analytic_gradient = False,
        compute_target_velocity =               False,
        compute_target_analytic_gradient =      False,
        compute_target_anti_analytic_gradient = False,
        array_source_velocity =                 None,
        array_source_analytic_gradient =        None,
        array_source_anti_analytic_gradient =   None,
        array_target_velocity =                 None,
        array_target_analytic_gradient =        None,
        array_target_anti_analytic_gradient =   None,
        precision = 4,
    ):
    """
    Pythonic interface for Biharmonic FMM
    Wraps the function:
        bfmm2dparttarg
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    charge      (optional), complex(ns):  charges at source locations
    dipole1     (optional), complex(ns):  dipole1 at source locations
    dipole2     (optional), complex(ns):  dipole2 at source locations
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), complex(n):   preallocated arrays for result
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'u'                        : complex(ns),   potential
            'u_analytic_gradient'      : complex(ns), analytic gradient
            'u_anti_analytic_gradient' : complex(ns), anti-analytic gradient
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            2) If array_#_* is provided and was acceptable, the code:
                "array_#_* is output['#']['*']"
                will return True
                If the array was provided but incorrect, then the code
                will return False
            3) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    charge, ifcharge = check_array(charge, (ns,), complex, 'charge')
    dipole1, ifdipole1 = check_array(dipole1, (ns,), complex, 'dipole1')
    dipole2, ifdipole2 = check_array(dipole2, (ns,), complex, 'dipole2')
    if (ifdipole1 or ifdipole2) and not (ifdipole1 and ifdipole2):
        raise Exception('If one of the dipoles is set, than the other must \
                             also be set.')
    ifdipole = ifdipole1
    vel, ifvel = check_output(array_source_velocity,
                        compute_source_velocity, (ns,), complex)
    grada, ifgrada = check_output(array_source_analytic_gradient,
                        compute_source_analytic_gradient, (ns,), complex)
    gradaa, ifgradaa = check_output(array_source_anti_analytic_gradient,
                        compute_source_anti_analytic_gradient, (ns,), complex)
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_velocity or compute_target_analytic_gradient \
             or compute_target_anti_analytic_gradient:
            raise Exception('If asking for a target quanitity, \
                    target must be given')
    veltarg, ifveltarg = check_output(array_target_velocity,
                        compute_target_velocity, (nt,), complex)
    gradatarg, ifgradatarg = check_output(array_target_analytic_gradient,
                        compute_target_analytic_gradient, (nt,), complex)
    gradaatarg, ifgradaatarg = check_output(array_target_anti_analytic_gradient,
                        compute_target_anti_analytic_gradient, (nt,), complex)
    ier = int(0)
    iprec = initialize_precision(precision)

    bhfmm2dparttarg(ier, iprec, ns, source, ifcharge, charge, ifdipole, dipole1, 
        dipole2, ifvel, vel, ifgrada, grada, ifgradaa, gradaa, nt, target,
        ifveltarg, veltarg, ifgradatarg, gradatarg, ifgradaatarg, gradaatarg)

    output = {}
    any_source = compute_source_velocity or compute_source_analytic_gradient \
                    or compute_source_anti_analytic_gradient
    if any_source:
        source_output = {}
        if compute_source_velocity:
            source_output['u'] = vel
        if compute_source_analytic_gradient:
            source_output['u_analytic_gradient'] = grada
        if compute_source_anti_analytic_gradient:
            source_output['u_anti_analytic_gradient'] = gradaa
        output['source'] = source_output
    any_target = compute_target_velocity or compute_target_analytic_gradient \
                    or compute_target_anti_analytic_gradient
    if any_target:
        target_output = {}
        if compute_target_velocity:
            target_output['u'] = veltarg
        if compute_target_analytic_gradient:
            target_output['u_analytic_gradient'] = gradatarg
        if compute_target_anti_analytic_gradient:
            target_output['u_anti_analytic_gradient'] = gradaatarg
        output['target'] = target_output
    output['ier'] = ier
    return output

def SFMM(
        source,
        target  = None,
        forces  = None,
        dipstr  = None,
        dipvec  = None,
        compute_source_velocity = False,
        compute_source_stress =   False,
        compute_target_velocity = False,
        compute_target_stress =   False,
        precision = 4,
    ):
    """
    Pythonic interface for Stokes FMM
    Wraps the function:
        bfmm2dparttarg
    
    Parameters:
    source      (required), float(2, ns): location of sources
    target      (optional), float(2, nt): location of targets
    forces      (optional), float(2, ns): forces at source locations
    dipstr      (optional), float(2, ns): dipole strengths at source locations
    dipvec      (optional), float(2, ns): orientation vector of dipoles
        if dipstr is set, then dipvec must be, also
    compute_#_* (optional), bool:         whether to compute * at # locations
    array_#_*   (optional), float(k,n): preallocated arrays for result
        k = 2 for *=velocity, k=5 for *=stress
        n = ns for #=source, nt for #=target
        if these arrays are not provided, are not of the correct size, not
            of the correct type, or not fortran contiguous, new arrays for
            the results will be allocated at runtime
    precision    (optional), float: precision, see documentation for FMM

    Returns:
    Dictionary:
        'ier': (integer) output code
            0:     successful completion of code
            4: failure to allocate memory for tree
            8: failure to allocate memory for FMM workspaces
            16: failure to allocate memory for multipole/local
                expansions
        'source': (quantities computed at source locations)
            'u'        : float(ns),   velocity, x-direction
            'v'        : float(ns),   velocity, y-direction
            'u_x'      : float(ns),   x derivaitive of u
            'u_y'      : float(ns),   y derivaitive of u
            'v_x'      : float(ns),   x derivaitive of v
            'v_y'      : float(ns),   y derivaitive of v
            'p'        : float(ns),   pressure
        'target': (quantities computed at target locations):
            same as above, but for target related things
            ns replaced by nt, in the shapes
        Some notes about the output:
            1) Entries of the dictionary will only exist if they were asked for
                i.e. if no 'source' quantities were requested, the 'source'
                dictionary will not exist
    """
    source, _, ns = check_array(source, (2,None), float, 'source', True)
    forces, ifforces = check_array(forces, (2, ns), float, 'forces')
    dipstr, ifdipstr = check_array(dipstr, (2, ns), float, 'dipstr')
    dipvec, ifdipvec = check_array(dipvec, (2, ns), float, 'dipvec')
    if ifdipstr and not ifdipvec:
        raise Exception('If dipstr is provided, dipvec must be also')
    # construct inputs for the biharmonic FMM
    sca = 1.0/(4.0*np.pi)
    if ifforces:
        tc = forces[0] + 1j*forces[1]
        cc = -0.5j*tc*sca
    else:
        cc = None
    if ifdipstr:
        dsc = 1j*(dipstr[0] + 1j*dipstr[1])
        dvc = dipvec[0] + 1j*dipvec[1]
        d1 = dsc*dvc*sca
        d2 = (dsc*np.conj(dvc) - np.conj(dsc)*dvc)*sca
    else:
        d1 = None
        d2 = None
    target, iftarget, nt = check_array(target, (2,None), float, 'target', True)
    if not iftarget:
        if compute_target_velocity or compute_target_stress:
            raise Exception('If asking for a target quanitity, \
                    target must be given')

    bout = BFMM(source,
                target, 
                cc,
                d1,
                d2,
                compute_source_velocity,
                compute_source_stress,
                compute_source_stress,
                compute_target_velocity,
                compute_target_stress,
                compute_target_stress)

    output = {}
    any_source = compute_source_velocity or compute_source_stress
    if any_source:
        source_output = {}
        if compute_source_velocity:
            vel = -1.0j*bout['source']['u']
            if ifforces:
                correction = np.sum(cc) - cc
                vel += 1j*correction
            source_output['u'] = vel.real
            source_output['v'] = vel.imag
        if compute_source_stress:
            agrad = bout['source']['u_analytic_gradient']
            aagrad = bout['source']['u_anti_analytic_gradient']
            w = -4*agrad.real
            w2 = -2*aagrad.real
            ux = aagrad.imag
            vy = -ux
            vx = 0.5*(w+w2)
            uy = vx - w
            p = -4*agrad.imag
            source_output['u_x'] = ux
            source_output['u_y'] = uy
            source_output['v_x'] = vx
            source_output['v_y'] = vy
        output['source'] = source_output
    any_target = compute_target_velocity or compute_target_stress
    if any_target:
        target_output = {}
        if compute_target_velocity:
            vel = -1.0j*bout['target']['u']
            if ifforces:
                vel += 1j*np.sum(cc)
            target_output['u'] = vel.real
            target_output['v'] = vel.imag
        if compute_target_stress:
            agrad = bout['target']['u_analytic_gradient']
            aagrad = bout['target']['u_anti_analytic_gradient']
            w = -4*agrad.real
            w2 = -2*aagrad.real
            ux = aagrad.imag
            vy = -ux
            vx = 0.5*(w+w2)
            uy = vx - w
            p = -4*agrad.imag
            target_output['u_x'] = ux
            target_output['u_y'] = uy
            target_output['v_x'] = vx
            target_output['v_y'] = vy
            target_output['p']   = p
        output['target'] = target_output
    return output

function_map = {
    'helmholtz'       : HFMM,
    'laplace-complex' : LFMM,
    'laplace-real'    : RFMM,
    'laplace'         : RFMM,
    'cauchy'          : ZFMM,
    'cauchy-general'  : CFMM,
    'biharmonic'      : BFMM,
    'stokes'          : SFMM,
}
