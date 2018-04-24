function [U]=rfmm2dpart(iprec,nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,ifpot,ifgrad,ifhess,ntarget,target,ifpottarg,ifgradtarg,ifhesstarg)
%RFMM2DPART Laplace particle target FMM in R^2 (real).
%
% Laplace FMM in R^2: evaluate all pairwise particle
% interactions (ignoring self-interactions) and interactions with targets.
%
% rfmm2d: charge and dipstr are real valued, x \in R^2
%
% \phi(x_i) = \sum_{j\ne i}   charge_j \log |x_i-x_j|  
%                    + dipstr_j (dipvec_j \dot \grad_j log|x_i-x_j|)
%
% or, more precisely,
%
% \phi(x_i) = \sum_{j\ne i}   charge_j \log |x_i-x_j|  
%                   + dipstr_j (dipvec_j \dot (x_i-x_j)) * (-1/|x_i-x_j|^2)
%
%
% [U]=RFMM2DPART(IPREC,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC);
%
% [U]=RFMM2DPART(IPREC,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC,IFPOT,IFGRAD,IFHESS);
%
% [U]=RFMM2DPART(IPREC,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC,IFPOT,IFGRAD,IFHESS,...
%         NTARGET,TARGET,IFPOTTARG,IFGRADTARG,IFHESSTARG);
%
%
% This subroutine evaluates the Laplace potential, gradient and hessian due
% to a collection of charges and dipoles. We use log(r) for the 
% Green's function. 
%
% Self-interactions are not-included.
%
% Input parameters:
% 
% iprec - FMM precision flag
%
%             -2 => tolerance =.5d0   =>  
%             -1 => tolerance =.5d-1  =>  1 digit 
%              0 => tolerance =.5d-2  =>  2 digits
%              1 => tolerance =.5d-3  =>  3 digits
%              2 => tolerance =.5d-6  =>  6 digits
%              3 => tolerance =.5d-9  =>  9 digits
%              4 => tolerance =.5d-12 => 12 digits
%              5 => tolerance =.5d-15 => 15 digits
%
% nsource - number of sources
% source - real (2,nsource): source locations
% ifcharge - charge computation flag
%
%         0 => do not compute
%         1 => include charge contribution
% 
% charge - real (nsource): charge strengths 
% ifdipole - dipole computation flag
%
%         0 => do not compute
%         1 => include dipole contributions
% 
% dipole - real (nsource): dipole strengths
% dipvec - real (2,source): dipole orientation vectors
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
% U.pot - real (nsource) - potential at source locations
% U.grad - real (2,nsource) - gradient  at source locations
% U.hess - real (3,nsource) - hessian at source locations
% U.pottarg - real (ntarget) - potential at target locations
% U.gradtarg - real (2,ntarget) - gradient  at target locations
% U.hesstarg - real (3,ntarget) - hessian at target locations
%
% U.ier - error return code
%
%             ier=0     =>  normal execution
%             ier=4     =>  cannot allocate tree workspace
%             ier=8     =>  cannot allocate bulk FMM  workspace
%             ier=16    =>  cannot allocate mpole expansion workspace in FMM
%

if( nargin == 9 ) 
  ifpot = 1;
  ifgrad = 1;
  ifhess = 1;
  ntarget = 0;
  target = zeros(2,1);
  ifpottarg = 0;
  ifgradtarg = 0;
  ifhesstarg = 0;
end

if( nargin == 11 ) 
  ntarget = 0;
  target = zeros(2,1);
  ifpottarg = 0;
  ifgradtarg = 0;
  ifhesstarg = 0;
end

if( nargin == 13 ) 
  ifpottarg = 1;
  ifgradtarg = 1;
  ifhesstarg = 1;
end


ifcharge = double(ifcharge); ifdipole = double(ifdipole);
ifpot = double(ifpot); ifgrad = double(ifgrad); ifhess = double(ifhess);
ifpottarg = double(ifpottarg); ifgradtarg = double(ifgradtarg);
ifhesstarg = double(ifhesstarg);

pot=0;
grad=zeros(2,1);
hess=zeros(3,1);
pottarg=0;
gradtarg=zeros(2,1);
hesstarg=zeros(3,1);

if( ifpot == 1 ), pot=zeros(1,nsource); end;
if( ifgrad == 1 ), grad=zeros(2,nsource); end;
if( ifhess == 1 ), hess=zeros(3,nsource); end;
if( ifpottarg == 1 ), pottarg=zeros(1,ntarget); end;
if( ifgradtarg == 1 ), gradtarg=zeros(2,ntarget); end;
if( ifhesstarg == 1 ), hesstarg=zeros(3,ntarget); end;

ier=0;

if( ntarget == 0 ) 
mex_id_ = 'rfmm2dpartself(io int[x], i int[x], i int[x], i double[xx], i int[x], i double[], i int[x], i double[], i double[xx], i int[x], io double[], i int[x], io double[], i int[x], io double[])';
[ier, pot, grad, hess] = fmm2d(mex_id_, ier, iprec, nsource, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, pot, ifgrad, grad, ifhess, hess, 1, 1, 1, 2, nsource, 1, 1, 2, nsource, 1, 1, 1);
else
mex_id_ = 'rfmm2dparttarg(io int[x], i int[x], i int[x], i double[xx], i int[x], i double[], i int[x], i double[], i double[xx], i int[x], io double[], i int[x], io double[], i int[x], io double[], i int[x], i double[], i int[x], io double[], i int[x], io double[], i int[x], io double[])';
[ier, pot, grad, hess, pottarg, gradtarg, hesstarg] = fmm2d(mex_id_, ier, iprec, nsource, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, pot, ifgrad, grad, ifhess, hess, ntarget, target, ifpottarg, pottarg, ifgradtarg, gradtarg, ifhesstarg, hesstarg, 1, 1, 1, 2, nsource, 1, 1, 2, nsource, 1, 1, 1, 1, 1, 1, 1);
end


if( ifpot == 1 ) U.pot=pot; end
if( ifgrad == 1 ) U.grad=grad; end
if( ifhess == 1 ) U.hess=hess; end
if( ifpottarg == 1 ) U.pottarg=pottarg; end
if( ifgradtarg == 1 ) U.gradtarg=gradtarg; end
if( ifhesstarg == 1 ) U.hesstarg=hesstarg; end
U.ier=ier;



