function [U]=c2dpartdirect(nsource,source,ifcharge,charge,ifdipole,dipstr,ifpot,ifgrad,ifhess,ntarget,target,ifpottarg,ifgradtarg,ifhesstarg)
%C2DPARTDIRECT Laplace interactions in R^2, direct evaluation (g. Cauchy sums).
%
% Laplace direct evaluation in R^2: evaluate all pairwise particle
% interactions (ignoring self-interactions) and interactions with targets.
%
% \phi(z_i) = \sum_{j\ne i}   charge_j \log(z_i-z_j)  + dipstr_j *(1/(z_i-z_j))
%
% \gradient \phi(z_i) = \frac{\partial \phi(z_i)}{\partial z}
% \hessian  \phi(z_i) = \frac{\partial^2 \phi(z_i)}{\partial z^2}
%
% where charge and dipstr are complex valued, z are complex numbers.
%
%
% [U]=C2DPARTDIRECT(NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR);
%
% [U]=C2DPARTDIRECT(NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,IFPOT,IFGRAD,IFHESS);
%
% [U]=C2DPARTDIRECT(NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,IFPOT,IFGRAD,IFHESS,...
%         NTARGET,TARGET,IFPOTTARG,IFGRADTARG,IFHESSTARG);
%
%
% This subroutine evaluates the Laplace potential and gradient due
% to a collection of charges and dipoles. We use log(r) for the 
% Green's function. Self-interactions are not-included.
%
% Input parameters:
% 
% nsource - number of sources
% source - real (2,nsource): source locations
% ifcharge - charge computation flag
%
%         0 => do not compute
%         1 => include charge contribution
% 
% charge - complex (nsource): charge strengths 
% ifdipole - dipole computation flag
%
%         0 => do not compute
%         1 => include dipole contributions
% 
% dipole - complex (nsource): dipole strengths
%
% ifpot - potential computation flag, 1 => compute the potential, otherwise no
% ifgrad - gradient computation flag, 1 => compute the gradient, otherwise no
% ifhess - hessian computation flag, 1 => compute the hessian, otherwise no
%
% ntarget - number of targets
% target - real (2,ntarget): target locations
%
% ifpottarg - target potential computation flag, 
%      1 => compute the target potential, otherwise no
% ifgradtarg - target gradient computation flag, 
%      1 => compute the target gradient, otherwise no
% ihesstarg - target hessian computation flag 
%      1 => compute the hessian, otherwise no
%
% Output parameters: 
%
% U.pot - complex (nsource) - potential at source locations
% U.grad - complex (nsource) - gradient  at source locations
% U.hess - complex (nsource) - hessian at source locations
% U.pottarg - complex (ntarget) - potential at target locations
% U.gradtarg - complex (ntarget) - gradient  at target locations
% U.hesstarg - complex (ntarget) - hessian at target locations
%
% U.ier - error return code
%
%             ier=0     =>  normal execution
%

if( nargin == 6 ) 
  ifpot = 1;
  ifgrad = 1;
  ifhess = 1;
  ntarget = 0;
  target = zeros(2,1);
  ifpottarg = 0;
  ifgradtarg = 0;
  ifhesstarg = 0;
end

if( nargin == 9 ) 
  ntarget = 0;
  target = zeros(2,1);
  ifpottarg = 0;
  ifgradtarg = 0;
  ifhesstarg = 0;
end

if( nargin == 11 ) 
  ifpottarg = 1;
  ifgradtarg = 1;
  ifhesstarg = 1;
end


ifcharge = double(ifcharge); ifdipole = double(ifdipole);
ifpot = double(ifpot); ifgrad = double(ifgrad); ifhess = double(ifhess);
ifpottarg = double(ifpottarg); ifgradtarg = double(ifgradtarg);
ifhesstarg = double(ifhesstarg);

pot=0;
grad=0;
hess=0;
pottarg=0;
gradtarg=0;
hesstarg=0;

if( ifpot == 1 ), pot=zeros(1,nsource)+1i*zeros(1,nsource); end;
if( ifgrad == 1 ), grad=zeros(2,nsource)+1i*zeros(2,nsource); end;
if( ifhess == 1 ), hess=zeros(3,nsource)+1i*zeros(3,nsource); end;
if( ifpottarg == 1 ), pottarg=zeros(1,ntarget)+1i*zeros(1,ntarget); end;
if( ifgradtarg == 1 ), gradtarg=zeros(2,ntarget)+1i*zeros(2,ntarget); end;
if( ifhesstarg == 1 ), hesstarg=zeros(3,ntarget)+1i*zeros(3,ntarget); end;

ier=0;

mex_id_ = 'c2dpartdirect(i int[x], i double[xx], i int[x], i dcomplex[], i int[x], i dcomplex[], i int[x], io dcomplex[], i int[x], io dcomplex[], i int[x], io dcomplex[], i int[x], i double[], i int[x], io dcomplex[], i int[x], io dcomplex[], i int[x], io dcomplex[])';
[pot, grad, hess, pottarg, gradtarg, hesstarg] = fmm2d(mex_id_, nsource, source, ifcharge, charge, ifdipole, dipstr, ifpot, pot, ifgrad, grad, ifhess, hess, ntarget, target, ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg, 1, 2, nsource, 1, 1, 1, 1, 1, 1, 1, 1, 1);


if( ifpot == 1 ) U.pot=pot; end
if( ifgrad == 1 ) U.grad=grad; end
if( ifhess == 1 ) U.hess=hess; end
if( ifpottarg == 1 ) U.pottarg=pottarg; end
if( ifgradtarg == 1 ) U.gradtarg=gradtarg; end
if( ifhesstarg == 1 ) U.hesstarg=hesstarg; end
U.ier=ier;



