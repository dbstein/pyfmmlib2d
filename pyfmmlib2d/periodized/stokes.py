import numpy as np
from pyfmmlib2d import SFMM
from pyfmmlib2d.periodized.shift_to_box import shift_to_box

def stokes_kernel(sx, tx, w):
    ns = sx.shape[1]
    nt = tx.shape[1]
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d = np.sqrt(d2)
    logr = np.log(d)
    c = 0.25/np.pi
    G00 = (-logr + dx*dx/d2)
    G01 = dx*dy/d2
    G11 = (-logr + dy*dy/d2)
    u_mat = np.zeros([nt, 2*ns], dtype=float)
    v_mat = np.zeros([nt, 2*ns], dtype=float)
    u_mat[:, 0*ns:1*ns] = c*G00*w
    u_mat[:, 1*ns:2*ns] = c*G01*w
    v_mat[:, 0*ns:1*ns] = c*G01*w
    v_mat[:, 1*ns:2*ns] = c*G11*w
    return u_mat, v_mat
def stokes_kernel_stress(sx, tx, w):
    ns = sx.shape[1]
    nt = tx.shape[1]
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2*d2
    # forces part, u_x
    c = 0.25/np.pi
    G00 = (dx*dy**2 - dx**3)/d4
    G01 = (dy**3 - dx**2*dy)/d4
    ux_mat = np.zeros([nt, 2*ns])
    ux_mat[:, 0*ns:1*ns] = c*G00*w
    ux_mat[:, 1*ns:2*ns] = c*G01*w
    # forces part, u_y
    G00 = (-3*dy*dx**2-dy**3)/d4
    G01 = (dx**3 - dx*dy**2)/d4
    uy_mat = np.zeros([nt, 2*ns])
    uy_mat[:, 0*ns:1*ns] = c*G00*w
    uy_mat[:, 1*ns:2*ns] = c*G01*w
    # forces part, v_x
    G01 = (dy**3 - dy*dx**2)/d4
    G11 = (-3*dx*dy**2-dx**3)/d4
    vx_mat = np.zeros([nt, 2*ns])
    vx_mat[:, 0*ns:1*ns] = c*G01*w
    vx_mat[:, 1*ns:2*ns] = c*G11*w
    # forces part, v_y
    G01 = (dx**3 - dy**2*dx)/d4
    G11 = (dy*dx**2 - dy**3)/d4
    vy_mat = np.zeros([nt, 2*ns])
    vy_mat[:, 0*ns:1*ns] = c*G01*w
    vy_mat[:, 1*ns:2*ns] = c*G11*w
    # forces part, p
    c = 0.5/np.pi
    G0 = dx/d2
    G1 = dy/d2
    p_mat = np.zeros([nt, 2*ns])
    p_mat[:, 0*ns:1*ns] = c*G0*w
    p_mat[:, 1*ns:2*ns] = c*G1*w
    return 2*ux_mat - p_mat, uy_mat + vx_mat, 2*vy_mat - p_mat

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

class periodized_stokes_fmm(object):
    def __init__(self, bounds=[0,2*np.pi,0,2*np.pi], p=16, N=4, expansion_factor=1.5, eps=1e-14):
        """
        Class to execute periodized Stokes FMM
        bounds: [xmin, xmax, ymin, ymax] (location of periodic box)
        p:      order of expansion to use on periodic walls
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
        self.check, self.check_weights, self.n_check = \
            generate_square(0.5*self.ranx, 0.5*self.rany, self.center, self.p, self.N)

        # get dumb check surface
        self.dumb_check, self.dumb_check_weights, _ = \
            generate_square(0.5*self.ranx*3, 0.5*self.rany*3, self.center, self.p, self.N)

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
        # generate source --> targ Stokes velocity matrix
        S2U, S2V = stokes_kernel(self.source, self.check, self.weights)
        # generate source --> targ Stokes stress matrix
        S2Sxx, S2Sxy, S2Syy = stokes_kernel_stress(self.source, self.check, self.weights)
        S2SNx = S2Sxx*self.normals[0][:,None] + S2Sxy*self.normals[1][:,None]
        S2SNy = S2Sxy*self.normals[0][:,None] + S2Syy*self.normals[1][:,None]
        # generate the full system that we'll have to solve
        self.MAT = np.zeros([2*self.n_check, 2*self.n_sources], dtype=float)
        # generate the full system that we'll have to solve
        p = self.p*self.N
        self.MAT[0*p:1*p] = S2U[1*p:2*p] - S2U[0*p:1*p]
        self.MAT[1*p:2*p] = S2U[3*p:4*p] - S2U[2*p:3*p]
        self.MAT[2*p:3*p] = S2V[1*p:2*p] - S2V[0*p:1*p]
        self.MAT[3*p:4*p] = S2V[3*p:4*p] - S2V[2*p:3*p]
        self.MAT[4*p:5*p] = S2SNx[1*p:2*p] - S2SNx[0*p:1*p]
        self.MAT[5*p:6*p] = S2SNx[3*p:4*p] - S2SNx[2*p:3*p]
        self.MAT[6*p:7*p] = S2SNy[1*p:2*p] - S2SNy[0*p:1*p]
        self.MAT[7*p:8*p] = S2SNy[3*p:4*p] - S2SNy[2*p:3*p]
        self.BIG_MAT = np.zeros([2*self.n_check+2, 2*self.n_sources+2], dtype=float)
        self.BIG_MAT[:-2,:-2] = self.MAT
        self.BIG_MAT[-2,0*self.n_sources:1*self.n_sources] = self.weights
        self.BIG_MAT[-1,1*self.n_sources:2*self.n_sources] = self.weights
        self.BIG_MAT[0*self.n_sources:1*self.n_sources,-2] = self.weights
        self.BIG_MAT[1*self.n_sources:2*self.n_sources,-1] = self.weights
        # take the SVD of this matrix
        self.U, D, self.VT = np.linalg.svd(self.BIG_MAT, full_matrices=False)
        D[D < eps] = np.Inf
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
        self.check_left_v   = np.zeros(NCL)
        self.check_right_v  = np.zeros(NCR)
        self.check_bottom_v = np.zeros(NCB)
        self.check_top_v    = np.zeros(NCT)
        self.check_left_snx   = np.zeros(NCL)
        self.check_right_snx  = np.zeros(NCR)
        self.check_bottom_snx = np.zeros(NCB)
        self.check_top_snx    = np.zeros(NCT)
        self.check_left_sny   = np.zeros(NCL)
        self.check_right_sny  = np.zeros(NCR)
        self.check_bottom_sny = np.zeros(NCB)
        self.check_top_sny    = np.zeros(NCT)
    def zero_check(self):
        self.check_left_u[:]     = 0.0
        self.check_right_u[:]    = 0.0
        self.check_bottom_u[:]   = 0.0
        self.check_top_u[:]      = 0.0
        self.check_left_v[:]     = 0.0
        self.check_right_v[:]    = 0.0
        self.check_bottom_v[:]   = 0.0
        self.check_top_v[:]      = 0.0
        self.check_left_snx[:]   = 0.0
        self.check_right_snx[:]  = 0.0
        self.check_bottom_snx[:] = 0.0
        self.check_top_snx[:]    = 0.0
        self.check_left_sny[:]   = 0.0
        self.check_right_sny[:]  = 0.0
        self.check_bottom_sny[:] = 0.0
        self.check_top_sny[:]    = 0.0
    def compute_to_check(self, src, fs, dps, dpv, left, right, bottom, top):
        if src.shape[1] > 0:
            out = SFMM(
                    source = src,
                    target = self.check,
                    forces = fs,
                    dipstr = dps,
                    dipvec = dpv,
                    compute_target_velocity = True,
                    compute_target_stress = True
                )
            u = out['target']['u']
            v = out['target']['v']
            ux = out['target']['u_x']
            uy = out['target']['u_y']
            vx = out['target']['v_x']
            vy = out['target']['v_y']
            p  = out['target']['p']
            sxx = 2*ux - p
            sxy = uy + vx
            syy = 2*vy - p
            if left:
                self.check_left_u += u[self.slice_left]
                self.check_left_v += v[self.slice_left]
                self.check_left_snx += sxx[self.slice_left]*self.normal_left[0]
                self.check_left_sny += sxy[self.slice_left]*self.normal_left[0]
            if right:
                self.check_right_u += u[self.slice_right]
                self.check_right_v += v[self.slice_right]
                self.check_right_snx += sxx[self.slice_right]*self.normal_right[0]
                self.check_right_sny += sxy[self.slice_right]*self.normal_right[0]
            if bottom:
                self.check_bottom_u += u[self.slice_bottom]
                self.check_bottom_v += v[self.slice_bottom]
                self.check_bottom_snx += sxy[self.slice_bottom]*self.normal_bottom[1]
                self.check_bottom_sny += syy[self.slice_bottom]*self.normal_bottom[1]
            if top:
                self.check_top_u += u[self.slice_top]
                self.check_top_v += v[self.slice_top]
                self.check_top_snx += sxy[self.slice_top]*self.normal_top[1]
                self.check_top_sny += syy[self.slice_top]*self.normal_top[1]

    def get_tau(self, tfx, tfy, force_sum, loc):
        # get jumps across the check surface
        ujumpx = self.check_right_u - self.check_left_u
        ujumpy = self.check_top_u   - self.check_bottom_u
        vjumpx = self.check_right_v - self.check_left_v
        vjumpy = self.check_top_v   - self.check_bottom_v
        snxjumpx = self.check_right_snx - self.check_left_snx + tfx/self.ranx
        snxjumpy = self.check_top_snx   - self.check_bottom_snx
        snyjumpx = self.check_right_sny - self.check_left_sny
        snyjumpy = self.check_top_sny   - self.check_bottom_sny + tfy/self.rany
        ujumps = np.concatenate([ujumpx, ujumpy, vjumpx, vjumpy, snxjumpx, snxjumpy, snyjumpx, snyjumpy])
        rhs = np.concatenate([-ujumps, -force_sum])
        tau = self.VT.T.dot(self.U.T.dot(rhs)*self.DI)[:-2]
        # weight by quadrature weights
        return tau.reshape(2, self.n_sources)*self.weights

    def sum_check(self, u, sl):
        w = self.check_weights[sl]
        return np.sum(u*w)/np.sum(w)

    def __internal_call__(self, source, tiling_distance, forces, dipstr, dipvec, loc):
        # get total forces
        tfx = np.sum(forces[0])
        tfy = np.sum(forces[1])
        # compute real distances
        dist_x = tiling_distance * self.ranx
        dist_y = tiling_distance * self.rany
        # zero the check surfaces
        self.zero_check()

        ########################################################################
        ###### location 'lb': lower bottom points
        if loc == 'lb':
            close_left = source[0] < self.bounds[0] + dist_x
            close_bottom = source[1] < self.bounds[2] + dist_y
            badx = close_left
            bady = close_bottom
            left = True
            bottom = True
        elif loc == 'lt':
            close_left = source[0] < self.bounds[0] + dist_x
            close_top = source[1] > self.bounds[3] - dist_y
            badx = close_left
            bady = close_top
            left = True
            bottom = False
        elif loc == 'rb':
            close_right = source[0] > self.bounds[1] - dist_x
            close_bottom = source[1] < self.bounds[2] + dist_y
            badx = close_right
            bady = close_bottom
            left = False
            bottom = True
        elif loc == 'rt':
            close_right = source[0] > self.bounds[1] - dist_x
            close_top = source[1] > self.bounds[3] - dist_y
            badx = close_right
            bady = close_top
            left = False
            bottom = False

        xsign = 1 if left else -1
        ysign = 1 if bottom else -1

        bad_locations = np.logical_or(badx, bady)
        good_locations = ~bad_locations
        close_xo = np.logical_and(badx, ~bady)
        close_yo = np.logical_and(~badx, bady)
        close_xy = np.logical_and(badx, bady)
        # tiling 1: points close only on x tiled over x
        close_xo_source = source[:,close_xo]
        close_xo_forces = forces[:,close_xo] if forces is not None else None
        close_xo_dipstr = dipstr[:,close_xo] if dipstr is not None else None
        close_xo_dipvec = dipvec[:,close_xo] if dipvec is not None else None
        # tile across x
        close_xo_tiled_source = np.row_stack([
                close_xo_source[0] + xsign*self.ranx,
                close_xo_source[1],
            ])
        # tiling 2: points close only on y tiled over y
        close_yo_source = source[:,close_yo]
        close_yo_forces = forces[:,close_yo] if forces is not None else None
        close_yo_dipstr = dipstr[:,close_yo] if dipstr is not None else None
        close_yo_dipvec = dipvec[:,close_yo] if dipvec is not None else None
        # tile across y
        close_yo_tiled_source = np.row_stack([
                close_yo_source[0],
                close_yo_source[1] + ysign*self.rany,
            ])
        # tiling 3: points close to x and y tiled over both
        close_xy_source = source[:,close_xy]
        close_xy_forces = forces[:,close_xy] if forces is not None else None
        close_xy_dipstr = dipstr[:,close_xy] if dipstr is not None else None
        close_xy_dipvec = dipvec[:,close_xy] if dipvec is not None else None
        # tile across x
        close_xy_xot_source = np.row_stack([
                close_xy_source[0] + xsign*self.ranx,
                close_xy_source[1],
            ])
        # tile across y
        close_xy_yot_source = np.row_stack([
                close_xy_source[0],
                close_xy_source[1] + ysign*self.rany,
            ])
        # tile across x and y
        close_xy_xyt_source = np.row_stack([
                close_xy_source[0] + xsign*self.ranx,
                close_xy_source[1] + ysign*self.rany,
            ])
        close_xy_tiled_source = np.column_stack([close_xy_xot_source, close_xy_yot_source, close_xy_xyt_source])
        close_xy_tiled_forces = np.tile(close_xy_forces,  (1,3)) if forces is not None else None
        close_xy_tiled_dipstr = np.tile(close_xy_dipstr,  (1,3)) if dipstr is not None else None
        close_xy_tiled_dipvec = np.tile(close_xy_dipvec,  (1,3)) if dipvec is not None else None

        # get new tiled sources for this call
        marginal_source = np.column_stack([close_xo_tiled_source, close_yo_tiled_source, close_xy_tiled_source])
        marginal_forces = np.column_stack([close_xo_forces,       close_yo_forces,       close_xy_tiled_forces]) if forces is not None else None
        marginal_dipstr = np.column_stack([close_xo_dipstr,       close_yo_dipstr,       close_xy_tiled_dipstr]) if dipstr is not None else None
        marginal_dipvec = np.column_stack([close_xo_dipvec,       close_yo_dipvec,       close_xy_tiled_dipvec]) if dipvec is not None else None
        # get all source / forces / dipstr / dipvec for this subcall
        all_source = np.column_stack([source, marginal_source ])
        all_forces = np.column_stack([forces, marginal_forces ]) if forces is not None else None
        all_dipstr = np.column_stack([dipstr, marginal_dipstr ]) if dipstr is not None else None
        all_dipvec = np.column_stack([dipvec, marginal_dipvec ]) if dipvec is not None else None
            
        # compute close x-only sources to appropriate checks
        self.compute_to_check(
                close_xo_source,
                close_xo_forces,
                close_xo_dipstr,
                close_xo_dipvec,
                left=not left, right=left, bottom=True, top=True,
            )
        # compute close x-tiled sources to appropriate checks
        self.compute_to_check(
                close_xo_tiled_source,
                close_xo_forces,
                close_xo_dipstr,
                close_xo_dipvec,
                left=left, right=not left, bottom=True, top=True,
            )
        # compute close y-only sources to appropriate checks
        self.compute_to_check(
                close_yo_source,
                close_yo_forces,
                close_yo_dipstr,
                close_yo_dipvec,
                left=True, right=True, bottom=not bottom, top=bottom,
            )
        # compute close y-only tiled sources to appropriate checks
        self.compute_to_check(
                close_yo_tiled_source,
                close_yo_forces,
                close_yo_dipstr,
                close_yo_dipvec,
                left=True, right=True, bottom=bottom, top=not bottom,
            )
        # compute close-xy sources to appropriate checks
        self.compute_to_check(
                close_xy_source,
                close_xy_forces,
                close_xy_dipstr,
                close_xy_dipvec,
                left=not left, right=left, bottom=not bottom, top=bottom,
            )
        # compute close-xy tiled sources to appropriate checks
        self.compute_to_check(
                close_xy_xot_source,
                close_xy_forces,
                close_xy_dipstr,
                close_xy_dipvec,
                left=left, right=not left, bottom=not bottom, top=bottom,
            )
        self.compute_to_check(
                close_xy_yot_source,
                close_xy_forces,
                close_xy_dipstr,
                close_xy_dipvec,
                left=not left, right=left, bottom=bottom, top=not bottom,
            )
        self.compute_to_check(
                close_xy_xyt_source,
                close_xy_forces,
                close_xy_dipstr,
                close_xy_dipvec,
                left=left, right=not left, bottom=bottom, top=not bottom,
            )
        # compute good sources to appropriate checks
        good_source = source[:,good_locations]
        good_forces = forces[:,good_locations] if forces is not None else None
        good_dipstr = dipstr[:,good_locations] if dipstr is not None else None
        good_dipvec = dipvec[:,good_locations] if dipvec is not None else None
        self.compute_to_check(
                good_source,
                good_forces,
                good_dipstr,
                good_dipvec,
                left=True, right=True, bottom=True, top=True
        )
        # get effective density
        tau = self.get_tau(tfx, tfy, np.sum(all_forces, axis=1), loc)
        # flux calculation
        slh_x = self.slice_right if left else self.slice_left
        flux_check_x = self.check[:,slh_x].copy()
        flux_check_x[0] -= xsign*0.25*self.ranx
        slh_y = self.slice_top if bottom else self.slice_bottom
        flux_check_y = self.check[:,slh_y].copy()
        flux_check_y[1] -= ysign*0.25*self.rany
        flux_check = np.column_stack([flux_check_x, flux_check_y])
        NH = flux_check_x.shape[1]
        # evaluate sources onto the flux check boundaries
        out3 = SFMM(
                    source = all_source,
                    target = flux_check,
                    forces = all_forces,
                    dipstr = all_dipstr,
                    dipvec = all_dipvec,
                    compute_target_velocity = True,
                )
        # evaluate tau onto the flux check boundaries
        out4 = SFMM(
                    source = self.source,
                    target = flux_check,
                    forces = tau,
                    compute_target_velocity = True,
                )
        flux_x = out3['target']['u'][:NH] + out4['target']['u'][:NH]
        flux_y = out3['target']['v'][NH:] + out4['target']['v'][NH:]
        flux_x = self.sum_check(flux_x, slh_x)
        flux_y = self.sum_check(flux_y, slh_y)

        return tau, flux_x, flux_y, marginal_source, marginal_forces, marginal_dipstr, marginal_dipvec

    def __call__(self, source, tiling_distance=0.15, forces=None, dipstr=None, dipvec=None, target=None, compute_source_velocity=False, compute_source_stress=False, compute_target_velocity=False, compute_target_stress=False):
        
        if compute_target_velocity or compute_target_stress:
            assert target is not None, 'Need to give target to compute target velocity or stresses'
        if target is None:
            target = np.zeros([2,1])

        # shift both the source and the target into box
        source = shift_to_box(source, self.bounds)
        target = shift_to_box(target, self.bounds) if target is not None else None

        # separate sources up into quadrants to allow clean flux compuations
        lefties = source[0] < self.center[0]
        righties = source[0] >= self.center[0]
        bottomies = source[1] < self.center[1]
        topies = source[1] >= self.center[1]
        lbs = np.logical_and(lefties, bottomies)
        lts = np.logical_and(lefties, topies)
        rbs = np.logical_and(righties, bottomies)
        rts = np.logical_and(righties, topies)
        # reduce to left-bottom sources
        lb_source = source[:,lbs]
        lb_forces = forces[:,lbs] if forces is not None else None
        lb_dipstr = dipstr[:,lbs] if dipstr is not None else None
        lb_dipvec = dipvec[:,lbs] if dipvec is not None else None
        # compute fluxes / density / tilings
        out_lb = self.__internal_call__(lb_source, tiling_distance, lb_forces, lb_dipstr, lb_dipvec, 'lb')
        # reduce to left-top sources
        lt_source = source[:,lts]
        lt_forces = forces[:,lts] if forces is not None else None
        lt_dipstr = dipstr[:,lts] if dipstr is not None else None
        lt_dipvec = dipvec[:,lts] if dipvec is not None else None
        # compute fluxes / density / tilings
        out_lt = self.__internal_call__(lt_source, tiling_distance, lt_forces, lt_dipstr, lt_dipvec, 'lt')
        # reduce to right-bottom sources
        rb_source = source[:,rbs]
        rb_forces = forces[:,rbs] if forces is not None else None
        rb_dipstr = dipstr[:,rbs] if dipstr is not None else None
        rb_dipvec = dipvec[:,rbs] if dipvec is not None else None
        # compute fluxes / density / tilings
        out_rb = self.__internal_call__(rb_source, tiling_distance, rb_forces, rb_dipstr, rb_dipvec, 'rb')
        # reduce to right-top sources
        rt_source = source[:,rts]
        rt_forces = forces[:,rts] if forces is not None else None
        rt_dipstr = dipstr[:,rts] if dipstr is not None else None
        rt_dipvec = dipvec[:,rts] if dipvec is not None else None
        # compute fluxes / density / tilings
        out_rt = self.__internal_call__(rt_source, tiling_distance, rt_forces, rt_dipstr, rt_dipvec, 'rt')

        # collect all sources / tiled sources / etc.
        full_source = np.column_stack([source, out_lb[3], out_lt[3], out_rb[3], out_rt[3]])
        full_forces = np.column_stack([forces, out_lb[4], out_lt[4], out_rb[4], out_rt[4]]) if forces is not None else None
        full_dipstr = np.column_stack([dipstr, out_lb[5], out_lt[5], out_rb[5], out_rt[5]]) if dipstr is not None else None
        full_dipvec = np.column_stack([dipvec, out_lb[6], out_lt[6], out_rb[6], out_rt[6]]) if dipvec is not None else None

        # collect all taus together
        tau = out_lb[0] + out_lt[0] + out_rb[0] + out_rt[0]
        # collect all fluxes together
        flux_x = out_lb[1] + out_lt[1] + out_rb[1] + out_rt[1]
        flux_y = out_lb[2] + out_lt[2] + out_rb[2] + out_rt[2]

        # compute FMM from tiled sources onto source and targets
        out1 = SFMM(
                    source = full_source,
                    target = target,
                    forces = full_forces,
                    dipstr = full_dipstr,
                    dipvec = full_dipvec,
                    compute_source_velocity = compute_source_velocity,
                    compute_source_stress   = compute_source_stress,
                    compute_target_velocity = compute_target_velocity,
                    compute_target_stress   = compute_target_stress,
                )
        # compute correction to set jumps to 0
        big_target = np.column_stack([ source, target ])
        out2 = SFMM(
                    source = self.source,
                    target = big_target,
                    forces = tau,
                    compute_target_velocity = compute_source_velocity or compute_target_velocity,
                    compute_target_stress   = compute_source_stress or compute_target_stress,
                )
        # add these together, along with fluxes, and return and return
        SN = source.shape[1]
        out = {}
        if compute_source_velocity or compute_source_stress:
            out_self = {}
            if compute_source_velocity:
                out_self['u'] = out1['source']['u'][:SN] + out2['target']['u'][:SN] - flux_x
                out_self['v'] = out1['source']['v'][:SN] + out2['target']['v'][:SN] - flux_y
            if compute_source_stress:
                out_self['u_x'] = out1['source']['u_x'][:SN] + out2['target']['u_x'][:SN]
                out_self['u_y'] = out1['source']['u_y'][:SN] + out2['target']['u_y'][:SN]
                out_self['v_x'] = out1['source']['v_x'][:SN] + out2['target']['v_x'][:SN]
                out_self['v_y'] = out1['source']['v_y'][:SN] + out2['target']['v_y'][:SN]
                out_self['p']   = out1['source']['p'][:SN]   + out2['target']['p'][:SN]
            out['self'] = out_self
        if compute_target_velocity or compute_target_stress:
            out_target = {}
            if compute_target_velocity:
                out_target['u'] = out1['target']['u'] + out2['target']['u'][SN:] - flux_x
                out_target['v'] = out1['target']['v'] + out2['target']['v'][SN:] - flux_y
            if compute_target_stress:
                out_target['u_x'] = out1['target']['u_x'] + out2['target']['u_x'][SN:]
                out_target['u_y'] = out1['target']['u_y'] + out2['target']['u_y'][SN:]
                out_target['v_x'] = out1['target']['v_x'] + out2['target']['v_x'][SN:]
                out_target['v_y'] = out1['target']['v_y'] + out2['target']['v_y'][SN:]
                out_target['p']   = out1['target']['p']   + out2['target']['p'][SN:]
            out['target'] = out_target
        return out
