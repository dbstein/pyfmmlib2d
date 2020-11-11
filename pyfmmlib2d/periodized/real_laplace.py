import numpy as np
import scipy as sp
import scipy.linalg
from pyfmmlib2d import RFMM

ipi2 = 0.5/np.pi

def laplace_kernel(sx, tx, w):
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d = np.hypot(dx, dy)
    return np.log(d)*ipi2*w
def laplace_kernel_gradient(sx, tx, w):
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = (dx**2 + dy**2)/ipi2
    return dx/d2*w, dy/d2*w

# quad weights for the chebyshev nodes i'm using
def fejer_1(n):
    points = -np.cos(np.pi * (np.arange(n) + 0.5) / n)
    N = np.arange(1, n, 2)
    length = len(N)
    m = n - length
    K = np.arange(m)
    v0 = np.concatenate(
        [
            2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K ** 2),
            np.zeros(length + 1),
        ]
    )
    v1 = v0[:-1] + np.conjugate(v0[:0:-1])
    w = np.fft.ifft(v1)
    weights = w.real
    return points, weights
def cheb(n, a=-1, b=1):
    p, w = fejer_1(n)
    ap = 0.5 * (p+1) * (b-a) + a
    return ap, 0.5*w*(b-a)

def generate_square(lx, ly, c, p, n):
    """
    Generate chebyshev discretization around square centered at c=(cx, cy)
    With width 2*lx and height 2*ly
    Discretization is chebyshev; with order p and number of panels n
    """
    ys = []
    yws = []
    xs = []
    xws = []
    wx = 2*lx/n
    wy = 2*ly/n
    for i in range(n):
        y, w = cheb(p, -ly + i*wy, -ly + (i+1)*wy)
        ys.append(y)
        yws.append(w)
        x, w = cheb(p, -lx + i*wx, -lx + (i+1)*wx)
        xs.append(x)
        xws.append(w)
    x = np.concatenate(xs)
    xw = np.concatenate(xws)
    y = np.concatenate(ys)
    yw = np.concatenate(yws)
    # stitch sides together to get nodes
    lxl = np.repeat(lx, p*n)
    lyl = np.repeat(ly, p*n)
    left   = np.row_stack([ -lxl + c[0],  y   + c[1] ])
    right  = np.row_stack([  lxl + c[0],  y   + c[1] ])
    bottom = np.row_stack([  x   + c[0], -lyl + c[1] ])
    top    = np.row_stack([  x   + c[0],  lyl + c[1] ])
    nodes = np.column_stack([ left, right, bottom, top ])
    # stitch together weights
    weights = np.concatenate([ yw, xw, yw, xw ])

    return nodes, weights, nodes.shape[1]

def generate_separated_square(lx, ly, c, p, n):
    """
    Generate chebyshev discretization around square centered at c=(cx, cy)
    With width 2*lx and height 2*ly
    Discretization is chebyshev; with order p and number of panels n
    """
    ys = []
    yws = []
    xs = []
    xws = []
    wx = 2*lx/n
    wy = 2*ly/n
    for i in range(n):
        y, w = cheb(p, -ly + i*wy, -ly + (i+1)*wy)
        ys.append(y)
        yws.append(w)
        x, w = cheb(p, -lx + i*wx, -lx + (i+1)*wx)
        xs.append(x)
        xws.append(w)
    x = np.concatenate(xs)
    xw = np.concatenate(xws)
    y = np.concatenate(ys)
    yw = np.concatenate(yws)
    # stitch sides together to get nodes
    lxl = np.repeat(lx, p*n)
    lyl = np.repeat(ly, p*n)
    left   = np.row_stack([ -lxl + c[0],  y   + c[1] ])
    right  = np.row_stack([  lxl + c[0],  y   + c[1] ])
    bottom = np.row_stack([  x   + c[0], -lyl + c[1] ])
    top    = np.row_stack([  x   + c[0],  lyl + c[1] ])

    return left, right, bottom, top

class periodized_laplace_fmm(object):
    def __init__(self, bounds=[0,2*np.pi,0,2*np.pi], p=16, N=4, expansion_factor=1.5, eps=1e-14):
        """
        Class to execute periodized FMM
        bounds: [xmin, xmax, ymin, ymax] (location of periodic box)
        p:      order of expansion to use on periodic walls
        N:      number of panels to use on each periodic wall
        """
        self.bounds = bounds
        self.p = p
        self.N = N
        self.ef = expansion_factor
        self.eps = eps
        # center of bounds
        self.center = [ 0.5*(self.bounds[0]+self.bounds[1]), 
                        0.5*(self.bounds[2]+self.bounds[3]) ]
        # ranges in both directions
        self.ranx = self.bounds[1] - self.bounds[0]
        self.rany = self.bounds[3] - self.bounds[2]
        if np.abs(self.ranx - self.rany) > 1e-15:
            raise Exception('For now, periodization bounds must be a square.')

        # get sources
        self.source, self.weights, self.n_sources = \
            generate_square(0.5*self.ranx*self.ef, 0.5*self.rany*self.ef, self.center, self.p, self.N)

        # get check surface
        self.check, _, self.n_check = \
            generate_square(0.5*self.ranx, 0.5*self.rany, self.center, self.p, self.N)

        self.check_left, self.check_right, self.check_bottom, self.check_top = \
            generate_separated_square(0.5*self.ranx, 0.5*self.rany, self.center, self.p, self.N)

        # get normals at check surface (not outward facing!)
        rep = lambda x: np.repeat(x, self.p*self.N)
        self.normal_left   = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_right  = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_bottom = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normal_top    = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normals = np.column_stack([ self.normal_left, self.normal_right,
                                         self.normal_bottom, self.normal_top ])

        # generate source --> targ Laplace pot matrix
        S2U = laplace_kernel(self.source, self.check, self.weights)
        # generate source --> targ Laplace grad matrix
        S2Gx, S2Gy = laplace_kernel_gradient(self.source, self.check, self.weights)
        S2N = S2Gx*self.normals[0][:,None] + S2Gy*self.normals[1][:,None]
        # generate the full system that we'll have to solve
        p = self.p*self.N
        self.MAT = np.zeros([self.n_sources, self.n_sources], dtype=float)
        self.MAT[0*p:1*p] = S2U[1*p:2*p] - S2U[0*p:1*p]
        self.MAT[1*p:2*p] = S2U[3*p:4*p] - S2U[2*p:3*p]
        self.MAT[2*p:3*p] = S2N[1*p:2*p] - S2N[0*p:1*p]
        self.MAT[3*p:4*p] = S2N[3*p:4*p] - S2N[2*p:3*p]
        # take the SVD of this matrix
        self.U, D, self.VT = np.linalg.svd(self.MAT, full_matrices=False)
        D[D < self.eps] = np.Inf
        self.DI = 1.0/D
        # setup slices for easy evaluations to check surfaces
        NCL = self.check_left.shape[1]
        NCR = self.check_right.shape[1]
        NCB = self.check_bottom.shape[1]
        NCT = self.check_top.shape[1]
        N0 = 0
        N1 = N0 + NCL
        N2 = N1 + NCR
        N3 = N2 + NCB
        N4 = N3 + NCT
        self.slice_left =   slice(N0, N1)
        self.slice_right =  slice(N1, N2)
        self.slice_bottom = slice(N2, N3)
        self.slice_top =    slice(N3, N4)
        # allocate space for check evaluations
        self.check_left_u   = np.zeros(NCL)
        self.check_right_u  = np.zeros(NCR)
        self.check_bottom_u = np.zeros(NCB)
        self.check_top_u    = np.zeros(NCT)
        self.check_left_dun   = np.zeros(NCL)
        self.check_right_dun  = np.zeros(NCR)
        self.check_bottom_dun = np.zeros(NCB)
        self.check_top_dun    = np.zeros(NCT)
    def zero_check(self):
        self.check_left_u[:]     = 0.0
        self.check_right_u[:]    = 0.0
        self.check_bottom_u[:]   = 0.0
        self.check_top_u[:]      = 0.0
        self.check_left_dun[:]   = 0.0
        self.check_right_dun[:]  = 0.0
        self.check_bottom_dun[:] = 0.0
        self.check_top_dun[:]    = 0.0
    def compute_to_check(self, src, ch, dps, dpv, left, right, bottom, top):
        out = RFMM(
                source = src,
                target = self.check,
                charge = ch,
                dipstr = dps,
                dipvec = dpv,
                compute_target_potential = True,
                compute_target_gradient = True
            )
        u = out['target']['u']
        ux = out['target']['u_x']
        uy = out['target']['u_y']
        if left:
            self.check_left_u += u[self.slice_left]
            self.check_left_dun += ux[self.slice_left]*self.normal_left[0]
        if right:
            self.check_right_u += u[self.slice_right]
            self.check_right_dun += ux[self.slice_right]*self.normal_right[0]
        if bottom:
            self.check_bottom_u += u[self.slice_bottom]
            self.check_bottom_dun += uy[self.slice_bottom]*self.normal_bottom[1]
        if top:
            self.check_top_u += u[self.slice_top]
            self.check_top_dun += uy[self.slice_top]*self.normal_top[1]
    def __call__(self, source, tiling_distance=0.15, charge=None, dipstr=None, dipvec=None, target=None, compute_source_potential=False, compute_source_gradient=False, compute_target_potential=False, compute_target_gradient=False):
        """
        Periodic Laplace FMM

        Reflection distance is how far to head to each side of the domain in order to tile (as percent of domain size...)
        """
        dist_x = tiling_distance * self.ranx
        dist_y = tiling_distance * self.rany
        if compute_target_potential or compute_target_gradient:
            assert target is not None, 'Need to give target to compute target potential or gradients'
        if target is None:
            target = np.zeros([2,1])
        # project charge onto solvability space
        charge = charge - np.mean(charge)
        # get where points are too close to boundary
        close_left = source[0] < self.bounds[0] + dist_x
        close_right = source[0] > self.bounds[1] - dist_x
        close_bottom = source[1] < self.bounds[2] + dist_y
        close_top = source[1] > self.bounds[3] - dist_y
        # get all bad locations
        bad_locations = np.logical_or.reduce([close_left, close_right, close_bottom, close_top])
        good_locations = ~bad_locations
        # further divide the close to boundary points
        far_bottom_top = ~np.logical_or(close_bottom, close_top)
        far_left_right = ~np.logical_or(close_left, close_right)
        close_left_bottom  = np.logical_and(close_left,   close_bottom)
        close_left_top     = np.logical_and(close_left,   close_top)
        close_left_only    = np.logical_and(close_left,   far_bottom_top)
        close_right_bottom = np.logical_and(close_right,  close_bottom)
        close_right_top    = np.logical_and(close_right,  close_top)
        close_right_only   = np.logical_and(close_right,  far_bottom_top)
        close_bottom_only  = np.logical_and(close_bottom, far_left_right)
        close_top_only     = np.logical_and(close_top,    far_left_right)
        # tiling 1: points close to only the left boundary tiled to the right
        left_only_bad_source = source[:,close_left_only]
        left_only_tiled_source = np.row_stack([
                left_only_bad_source[0] + self.ranx,
                left_only_bad_source[1],
            ])
        left_only_tiled_charge = charge[close_left_only]   if charge is not None else None
        left_only_tiled_dipstr = dipstr[close_left_only]   if dipstr is not None else None
        left_only_tiled_dipvec = dipvec[:,close_left_only] if dipvec is not None else None
        # tiling 2: points close to only the right boundary tiled to the left
        right_only_bad_source = source[:,close_right_only]
        right_only_tiled_source = np.row_stack([
                right_only_bad_source[0] - self.ranx,
                right_only_bad_source[1],
            ])
        right_only_tiled_charge = charge[close_right_only]   if charge is not None else None
        right_only_tiled_dipstr = dipstr[close_right_only]   if dipstr is not None else None
        right_only_tiled_dipvec = dipvec[:,close_right_only] if dipvec is not None else None
        # tiling 3: points close to only the bottom boundary tiled to the top
        bottom_only_bad_source = source[:,close_bottom_only]
        bottom_only_tiled_source = np.row_stack([
                bottom_only_bad_source[0],
                bottom_only_bad_source[1] + self.rany,
            ])
        bottom_only_tiled_charge = charge[close_bottom_only]   if charge is not None else None
        bottom_only_tiled_dipstr = dipstr[close_bottom_only]   if dipstr is not None else None
        bottom_only_tiled_dipvec = dipvec[:,close_bottom_only] if dipvec is not None else None
        # tiling 4: points close to only the top boundary tiled to  the bottom
        top_only_bad_source = source[:,close_top_only]
        top_only_tiled_source = np.row_stack([
                top_only_bad_source[0],
                top_only_bad_source[1] - self.rany,
            ])
        top_only_tiled_charge = charge[close_top_only]   if charge is not None else None
        top_only_tiled_dipstr = dipstr[close_top_only]   if dipstr is not None else None
        top_only_tiled_dipvec = dipvec[:,close_top_only] if dipvec is not None else None
        # tiling 5: points close to the left and the bottom boundaries
        left_bottom_bad_source = source[:,close_left_bottom]
        left_bottom_rb = np.row_stack([
                left_bottom_bad_source[0] + self.ranx,
                left_bottom_bad_source[1],
            ])
        left_bottom_rt = np.row_stack([
                left_bottom_bad_source[0] + self.ranx,
                left_bottom_bad_source[1] + self.rany,
            ])
        left_bottom_lt = np.row_stack([
                left_bottom_bad_source[0],
                left_bottom_bad_source[1] + self.rany,
            ])
        left_bottom_tiled_source = np.column_stack([left_bottom_rb, left_bottom_rt, left_bottom_lt])
        left_bottom_tiled_charge = charge[close_left_bottom]   if charge is not None else None
        left_bottom_tiled_dipstr = dipstr[close_left_bottom]   if dipstr is not None else None
        left_bottom_tiled_dipvec = dipvec[:,close_left_bottom] if dipvec is not None else None
        # tiling 6: points close to the left and the top boundaries, and their tilings
        left_top_bad_source = source[:,close_left_top]
        left_top_rt = np.row_stack([
                left_top_bad_source[0] + self.ranx,
                left_top_bad_source[1],
            ])
        left_top_rb = np.row_stack([
                left_top_bad_source[0] + self.ranx,
                left_top_bad_source[1] - self.rany,
            ])
        left_top_lb = np.row_stack([
                left_top_bad_source[0],
                left_top_bad_source[1] - self.rany,
            ])
        left_top_tiled_source = np.column_stack([left_top_rt, left_top_rb, left_top_lb])
        left_top_tiled_charge = charge[close_left_top]   if charge is not None else None
        left_top_tiled_dipstr = dipstr[close_left_top]   if dipstr is not None else None
        left_top_tiled_dipvec = dipvec[:,close_left_top] if dipvec is not None else None
        # tiling 7: points close to the right and the top boundaries
        right_top_bad_source = source[:,close_right_top]
        right_top_lt = np.row_stack([
                right_top_bad_source[0] - self.ranx,
                right_top_bad_source[1],
            ])
        right_top_lb = np.row_stack([
                right_top_bad_source[0] - self.ranx,
                right_top_bad_source[1] - self.rany,
            ])
        right_top_rb = np.row_stack([
                right_top_bad_source[0],
                right_top_bad_source[1] - self.rany,
            ])
        right_top_tiled_source = np.column_stack([right_top_lt, right_top_lb, right_top_rb])
        right_top_tiled_charge = charge[close_right_top]   if charge is not None else None
        right_top_tiled_dipstr = dipstr[close_right_top]   if dipstr is not None else None
        right_top_tiled_dipvec = dipvec[:,close_right_top] if dipvec is not None else None
        # tiling 8: points close to the right and the bottom boundaries
        right_bottom_bad_source = source[:,close_right_bottom]
        right_bottom_lb = np.row_stack([
                right_bottom_bad_source[0] - self.ranx,
                right_bottom_bad_source[1],
            ])
        right_bottom_lt = np.row_stack([
                right_bottom_bad_source[0] - self.ranx,
                right_bottom_bad_source[1] + self.rany,
            ])
        right_bottom_rt = np.row_stack([
                right_bottom_bad_source[0],
                right_bottom_bad_source[1] + self.rany,
            ])
        right_bottom_tiled_source = np.column_stack([right_bottom_lb, right_bottom_lt, right_bottom_rt])
        right_bottom_tiled_charge = charge[close_right_bottom]   if charge is not None else None
        right_bottom_tiled_dipstr = dipstr[close_right_bottom]   if dipstr is not None else None
        right_bottom_tiled_dipvec = dipvec[:,close_right_bottom] if dipvec is not None else None
        # add these into the "all left only" grouping
        all_left_only_source = np.column_stack([ left_only_bad_source,   right_only_tiled_source ])
        all_left_only_charge = np.concatenate([  left_only_tiled_charge, right_only_tiled_charge ]) if charge is not None else None
        all_left_only_dipstr = np.concatenate([  left_only_tiled_dipstr, right_only_tiled_dipstr ]) if dipstr is not None else None
        all_left_only_dipvec = np.column_stack([ left_only_tiled_dipvec, right_only_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "all right only" grouping
        all_right_only_source = np.column_stack([ right_only_bad_source,   left_only_tiled_source ])
        all_right_only_charge = np.concatenate([  right_only_tiled_charge, left_only_tiled_charge ]) if charge is not None else None
        all_right_only_dipstr = np.concatenate([  right_only_tiled_dipstr, left_only_tiled_dipstr ]) if dipstr is not None else None
        all_right_only_dipvec = np.column_stack([ right_only_tiled_dipvec, left_only_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "all bottom only" grouping
        all_bottom_only_source = np.column_stack([ bottom_only_bad_source,   top_only_tiled_source ])
        all_bottom_only_charge = np.concatenate([  bottom_only_tiled_charge, top_only_tiled_charge ]) if charge is not None else None
        all_bottom_only_dipstr = np.concatenate([  bottom_only_tiled_dipstr, top_only_tiled_dipstr ]) if dipstr is not None else None
        all_bottom_only_dipvec = np.column_stack([ bottom_only_tiled_dipvec, top_only_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "all top only" grouping
        all_top_only_source = np.column_stack([ top_only_bad_source,   bottom_only_tiled_source ])
        all_top_only_charge = np.concatenate([  top_only_tiled_charge, bottom_only_tiled_charge ]) if charge is not None else None
        all_top_only_dipstr = np.concatenate([  top_only_tiled_dipstr, bottom_only_tiled_dipstr ]) if dipstr is not None else None
        all_top_only_dipvec = np.column_stack([ top_only_tiled_dipvec, bottom_only_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "left bottom" grouping
        all_left_bottom_source = np.column_stack([ left_bottom_bad_source,   left_top_lb,               right_top_lb,               right_bottom_lb,              ])
        all_left_bottom_charge = np.concatenate([  left_bottom_tiled_charge, left_top_tiled_charge, right_top_tiled_charge, right_bottom_tiled_charge ]) if charge is not None else None
        all_left_bottom_dipstr = np.concatenate([  left_bottom_tiled_dipstr, left_top_tiled_dipstr, right_top_tiled_dipstr, right_bottom_tiled_dipstr ]) if dipstr is not None else None
        all_left_bottom_dipvec = np.column_stack([ left_bottom_tiled_dipvec, left_top_tiled_dipvec, right_top_tiled_dipvec, right_bottom_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "right bottom" grouping
        all_right_bottom_source = np.column_stack([ right_bottom_bad_source,   right_top_rb,               left_top_rb,               left_bottom_rb,              ])
        all_right_bottom_charge = np.concatenate([  right_bottom_tiled_charge, right_top_tiled_charge, left_top_tiled_charge, left_bottom_tiled_charge ]) if charge is not None else None
        all_right_bottom_dipstr = np.concatenate([  right_bottom_tiled_dipstr, right_top_tiled_dipstr, left_top_tiled_dipstr, left_bottom_tiled_dipstr ]) if dipstr is not None else None
        all_right_bottom_dipvec = np.column_stack([ right_bottom_tiled_dipvec, right_top_tiled_dipvec, left_top_tiled_dipvec, left_bottom_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "right top" grouping
        all_right_top_source = np.column_stack([ right_top_bad_source,   left_top_rt,               left_bottom_rt,               right_bottom_rt,              ])
        all_right_top_charge = np.concatenate([  right_top_tiled_charge, left_top_tiled_charge, left_bottom_tiled_charge, right_bottom_tiled_charge ]) if charge is not None else None
        all_right_top_dipstr = np.concatenate([  right_top_tiled_dipstr, left_top_tiled_dipstr, left_bottom_tiled_dipstr, right_bottom_tiled_dipstr ]) if dipstr is not None else None
        all_right_top_dipvec = np.column_stack([ right_top_tiled_dipvec, left_top_tiled_dipvec, left_bottom_tiled_dipvec, right_bottom_tiled_dipvec ]) if dipvec is not None else None
        # add these into the "left top" grouping
        all_left_top_source = np.column_stack([ left_top_bad_source,   left_bottom_lt,               right_bottom_lt,               right_top_lt,              ])
        all_left_top_charge = np.concatenate([  left_top_tiled_charge, left_bottom_tiled_charge, right_bottom_tiled_charge, right_top_tiled_charge ]) if charge is not None else None
        all_left_top_dipstr = np.concatenate([  left_top_tiled_dipstr, left_bottom_tiled_dipstr, right_bottom_tiled_dipstr, right_top_tiled_dipstr ]) if dipstr is not None else None
        all_left_top_dipvec = np.column_stack([ left_top_tiled_dipvec, left_bottom_tiled_dipvec, right_bottom_tiled_dipvec, right_top_tiled_dipvec ]) if dipvec is not None else None
        # sources / charges / dipstrs / dipvecs for direct portion of FMM
        lbr_sr = left_bottom_tiled_source
        rbr_sr = right_bottom_tiled_source
        rtr_sr = right_top_tiled_source
        ltr_sr = left_top_tiled_source
        lbr_ch = np.tile(left_bottom_tiled_charge, 3)      if charge is not None else None
        rbr_ch = np.tile(right_bottom_tiled_charge, 3)     if charge is not None else None
        rtr_ch = np.tile(right_top_tiled_charge, 3)        if charge is not None else None
        ltr_ch = np.tile(left_top_tiled_charge, 3)         if charge is not None else None
        lbr_ds = np.tile(left_bottom_tiled_dipstr, 3)      if dipstr is not None else None
        rbr_ds = np.tile(right_bottom_tiled_dipstr, 3)     if dipstr is not None else None
        rtr_ds = np.tile(right_top_tiled_dipstr, 3)        if dipstr is not None else None
        ltr_ds = np.tile(left_top_tiled_dipstr, 3)         if dipstr is not None else None
        lbr_dv = np.tile(left_bottom_tiled_dipvec, (1,3))  if dipvec is not None else None
        rbr_dv = np.tile(right_bottom_tiled_dipvec, (1,3)) if dipvec is not None else None
        rtr_dv = np.tile(right_top_tiled_dipvec, (1,3))    if dipvec is not None else None
        ltr_dv = np.tile(left_top_tiled_dipvec, (1,3))     if dipvec is not None else None
        all_source = np.column_stack([ source, left_only_tiled_source, right_only_tiled_source, bottom_only_tiled_source, top_only_tiled_source, lbr_sr, rbr_sr, rtr_sr, ltr_sr ])
        all_charge = np.concatenate([  charge, left_only_tiled_charge, right_only_tiled_charge, bottom_only_tiled_charge, top_only_tiled_charge, lbr_ch, rbr_ch, rtr_ch, ltr_ch ]) if charge is not None else None
        all_dipstr = np.concatenate([  dipstr, left_only_tiled_dipstr, right_only_tiled_dipstr, bottom_only_tiled_dipstr, top_only_tiled_dipstr, lbr_ds, rbr_ds, rtr_ds, ltr_ds ]) if dipstr is not None else None
        all_dipvec = np.column_stack([ dipvec, left_only_tiled_dipvec, right_only_tiled_dipvec, bottom_only_tiled_dipvec, top_only_tiled_dipvec, lbr_dv, rbr_dv, rtr_dv, ltr_dv ]) if dipvec is not None else None
        # set check evaluations to 0
        self.zero_check()
        # now compute things to appropriate portions of the check surfaces
        # first for the left-only sources
        self.compute_to_check(
                all_left_only_source,
                all_left_only_charge,
                all_left_only_dipstr,
                all_left_only_dipvec,
                left=False, right=True, bottom=True, top=True,
            )
        # now for the right-only sources
        self.compute_to_check(
                all_right_only_source,
                all_right_only_charge,
                all_right_only_dipstr,
                all_right_only_dipvec,
                left=True, right=False, bottom=True, top=True,
            )
        # now for the bottom-only sources
        self.compute_to_check(
                all_bottom_only_source,
                all_bottom_only_charge,
                all_bottom_only_dipstr,
                all_bottom_only_dipvec,
                left=True, right=True, bottom=False, top=True,
            )
        # now for the top-only sources
        self.compute_to_check(
                all_top_only_source,
                all_top_only_charge,
                all_top_only_dipstr,
                all_top_only_dipvec,
                left=True, right=True, bottom=True, top=False,
            )
        # now for the left-bottom sources
        self.compute_to_check(
                all_left_bottom_source,
                all_left_bottom_charge,
                all_left_bottom_dipstr,
                all_left_bottom_dipvec,
                left=False, right=True, bottom=False, top=True,
            )
        # now for the right-bottom sources
        self.compute_to_check(
                all_right_bottom_source,
                all_right_bottom_charge,
                all_right_bottom_dipstr,
                all_right_bottom_dipvec,
                left=True, right=False, bottom=False, top=True,
            )
        # now for the right-top sources
        self.compute_to_check(
                all_right_top_source,
                all_right_top_charge,
                all_right_top_dipstr,
                all_right_top_dipvec,
                left=True, right=False, bottom=True, top=False,
            )
        # now for the left-top sources
        self.compute_to_check(
                all_left_top_source,
                all_left_top_charge,
                all_left_top_dipstr,
                all_left_top_dipvec,
                left=False, right=True, bottom=True, top=False,
            )
        # now for the good sources
        good_source = source[:,good_locations]
        good_charge = charge[good_locations]   if charge is not None else None
        good_dipstr = dipstr[good_locations]   if dipstr is not None else None
        good_dipvec = dipvec[:,good_locations] if dipvec is not None else None
        self.compute_to_check(
                good_source,
                good_charge,
                good_dipstr,
                good_dipvec,
                left=True, right=True, bottom=True, top=True
            )

        # get jumps across the check surface
        ujumpx = self.check_right_u - self.check_left_u
        ujumpy = self.check_top_u   - self.check_bottom_u
        unjumpx = self.check_right_dun - self.check_left_dun
        unjumpy = self.check_top_dun   - self.check_bottom_dun
        ujumps = np.concatenate([ujumpx, ujumpy, unjumpx, unjumpy])
        # solve for sources that set these jumps to 0
        tau = -self.VT.T.dot(self.U.T.dot(ujumps)*self.DI)
        # compute FMM of tiled sources to source and targets
        out1 = RFMM(
                    source = all_source,
                    target = target,
                    charge = all_charge,
                    dipstr = all_dipstr,
                    dipvec = all_dipvec,
                    compute_source_potential = compute_source_potential,
                    compute_source_gradient  = compute_source_gradient,
                    compute_target_potential = compute_target_potential,
                    compute_target_gradient  = compute_target_gradient,
                )
        # compute correction to set jumps to 0
        big_target = np.column_stack([source, target])
        out2 = RFMM(
                    source = self.source,
                    target = big_target,
                    charge = tau*ipi2*self.weights,
                    compute_target_potential = compute_source_potential or compute_target_potential,
                    compute_target_gradient  = compute_source_gradient or compute_target_gradient,
                )
        # now add the tiling FMM to the correction FMM
        SN = source.shape[1]
        out = {}
        if compute_source_potential or compute_source_gradient:
            out_self = {}
            if compute_source_potential:
                out_self['u'] = out1['source']['u'][:SN] + out2['target']['u'][:SN]
            if compute_source_gradient:
                out_self['Du'] = out1['source']['Du'][:,:SN] + out2['target']['Du'][:,:SN]
                out_self['u_x'] = out_self['Du'][0]
                out_self['u_y'] = out_self['Du'][1]
            out['self'] = out_self
        if compute_target_potential or compute_target_gradient:
            out_target = {}
            if compute_target_potential:
                out_target['u'] = out1['target']['u'] + out2['target']['u'][SN:]
            if compute_target_gradient:
                out_target['Du'] = out1['target']['Du'] + out2['target']['Du'][:,SN:]
                out_target['u_x'] = out_target['Du'][0]
                out_target['u_y'] = out_target['Du'][1]
            out['target'] = out_target
        return out

