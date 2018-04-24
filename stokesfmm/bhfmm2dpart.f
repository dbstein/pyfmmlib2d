cc Copyright (C) 2010-2011: Leslie Greengard and Manas Rachh
cc Contact: rachh@cims.nyu.edu
cc 
cc This program is free software; you can redistribute it and/or modify 
cc it under the terms of the GNU General Public License as published by 
cc the Free Software Foundation; either version 2 of the License, or 
cc (at your option) any later version.  This program is distributed in 
cc the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
cc even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
cc PARTICULAR PURPOSE.  See the GNU General Public License for more 
cc details. You should have received a copy of the GNU General Public 
cc License along with this program; 
cc if not, see <http://www.gnu.org/licenses/>.
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c    $Date$
c    $Revision$
c    This file contains the main FMM routines and some related
c    subroutines for evaluating biharmonic sums due to
c    point "charges" and "dipoles". 
c
c    bhfmm2d does the biharmonic type sums of complex valued
c    charges and dipoles
c
c    bhfmm2d: charge and dip1,dip2 are complex valued,
c    z are complex numbers. The complex velocity for stokes/
c    displacement for stokes/elasticity problem take the general form 
c
c    Stokes:
c    vel(z_i) = \sum_{j \neq i} charge_j *\log|z_i-z_j| + 
c                (charge_j)_bar (z_i-z_j)/(z_i-z_j)_bar+
c                dip1/(z_i-z_j) + dip2/(z_i-z_j)_bar-
c                dip1_bar (z_i-z_j)/(z_i-z_j)^2_bar
c
c    The velocity for stokes is related to the goursat functions
c    as vel = \phi(z) + z (d/dz (\phi))_bar+\psi(z)
c 
c   The goursat functions are given by
c
c   \phi(z_i) = \sum_{j\neq i} charge_j log(z_i-z_j)+
c                     dip1/(z_i-z_j)
c
c   \psi(z_i) = \sum_{j\neq i} dippar2_bar/(z_i-z_j)+
c               z_j_bar dippar1/(z_i-z_j)^2 +
c               charge_j_bar log(z_i-z_j)- z_j_bar charge_j/(z_i-z_j)
c
c   For stokes and elasticity problems the following derivatives are
c   relevant to physical quantities such as pressure and vorticity 
c   for stokes and stresses for elasticity
c
c   1) Analytic gradient (grada) 
c      =  \frac{\partial \phi(z_i)}{\partial z}
c
c   2) Anti analytic gradient (gradaa)
c       = z \frac{\partial^2 \phi_(z_i)} {\partial^2 z}_bar +
c         \frac{\partial \psi_(z_i)} {\partial z} 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c   bhfmm2dpart - Generalized biharmonic FMM in R^2: 
c   evaluate all pairwise particle interactions (ignoring self-interaction)
c
c   bhfmm2dpartself - Generalized biharmonic FMM in R^2: 
c   evaluate all pairwise particle interactions (ignoring self-interaction)
c
c   bhfmm2dparttarg - Generalized biharmonic FMM in R^2: 
c   evaluate all pairwise particle interactions (ignoring self-interaction)
c   + interactions with targets
c
c   bh2dpartdirect - Generalized biharmonic interactions in R^2:  
c   evaluate all pairwise particle interactions (ignoring self-interaction) 
c   + interactions with targets via direct O(N^2) algorithm
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
        subroutine bhfmm2dpart(ier,iprec,nsource,source,
     $     ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa)
        implicit real *8 (a-h,o-z)
c              
c   Generalized biharmonic FMM in R^2: evaluate all pairwise particle
c   interactions (ignoring self-interaction). 
c   
c   bhfmm2d: charge and dipstr are complex valued, z are complex numbers.
c
c   bhfmm2d: charge and dip1,dip2 are complex valued,
c   z are complex numbers. The complex velocity for stokes/
c   displacement for stokes/elasticity problem take the general form 
c
c   Stokes:
c   vel(z_i) = \sum_{j \neq i} charge_j *\log|z_i-z_j| + 
c                (charge_j)_bar (z_i-z_j)/(z_i-z_j)_bar+
c                dip1/(z_i-z_j) + dip2/(z_i-z_j)_bar-
c                dip1_bar (z_i-z_j)/(z_i-z_j)^2_bar
c
c   The velocity for stokes is related to the goursat functions
c   as vel = \phi(z) + z (d/dz (\phi))_bar+\psi(z)
c 
c   The goursat functions are given by
c
c   \phi(z_i) = \sum_{j\neq i} charge_j log(z_i-z_j)+
c                     dip1/(z_i-z_j)
c
c   \psi(z_i) = \sum_{j\neq i} dippar2_bar/(z_i-z_j)+
c               z_j_bar dippar1/(z_i-z_j)^2 +
c               charge_j_bar log(z_i-z_j)- z_j_bar charge_j/(z_i-z_j)
c
c   For stokes and elasticity problems the following derivatives are
c   relevant to physical quantities such as pressure and vorticity 
c   for stokes and stresses for elasticity
c
c   1) Analytic gradient (grada) 
c      =  \frac{\partial \phi(z_i)}{\partial z}
c
c   2) Anti analytic gradient (gradaa)
c       = z \frac{\partial^2 \phi_(z_i)} {\partial^2 z}_bar +
c         \frac{\partial \psi_(z_i)} {\partial z}
c 
c   The main FMM routine permits both evaluation at sources
c   and at a collection of targets. 
c   This subroutine is used to simplify the user interface 
c   (by setting the number of targets to zero) and calling the more 
c   general FMM.
c
c   See below for explanation of calling sequence arguments.
c  
        real *8 source(2,*)
        complex *16 charge(*)
        complex *16 dip1(*)
        complex *16 dip2(*)
        complex *16 ima
        complex *16 vel(*)
        complex *16 grada(*)
        complex *16 gradaa(*)
c
        real *8 target(2,1)
        complex *16 veltarg(1)
        complex *16 gradatarg(1)
        complex *16 gradaatarg(1)        
c
        data ima/(0.0d0,1.0d0)/
c       
        ntarget=0
        ifveltarg=0
        ifgradatarg=0
        ifgradaatarg=0
c
        call bhfmm2dparttarg(ier,iprec,nsource,source,
     $  ifcharge,charge,ifdipole,dip1,dip2,
     $  ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $  ntarget,target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $  ifgradaatarg,gradaatarg)
c
        return
        end
c
c*************************************************************************
        subroutine bhfmm2dpartself(ier,iprec,nsource,source,
     $     ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa)
        implicit real *8 (a-h,o-z)
c              
c   Generalized biharmonic FMM in R^2: evaluate all pairwise particle
c   interactions (ignoring self-interaction). 
c
c   Self-interactions are not included.
c   
c   bhfmm2d: charge and dipstr are complex valued, z are complex numbers.
c
c   bhfmm2d: charge and dip1,dip2 are complex valued,
c   z are complex numbers. The complex velocity for stokes/
c   displacement for stokes/elasticity problem take the general form 
c
c   Stokes:
c   vel(z_i) = \sum_{j \neq i} charge_j *\log|z_i-z_j| + 
c                (charge_j)_bar (z_i-z_j)/(z_i-z_j)_bar+
c                dip1/(z_i-z_j) + dip2/(z_i-z_j)_bar-
c                dip1_bar (z_i-z_j)/(z_i-z_j)^2_bar
c
c   The velocity for stokes is related to the goursat functions
c   as vel = \phi(z) + z (d/dz (\phi))_bar+\psi(z)
c 
c   The goursat functions are given by
c
c   \phi(z_i) = \sum_{j\neq i} charge_j log(z_i-z_j)+
c                     dip1/(z_i-z_j)
c
c   \psi(z_i) = \sum_{j\neq i} dippar2_bar/(z_i-z_j)+
c               z_j_bar dippar1/(z_i-z_j)^2 +
c               charge_j_bar log(z_i-z_j)- z_j_bar charge_j/(z_i-z_j)
c
c   For stokes and elasticity problems the following derivatives are
c   relevant to physical quantities such as pressure and vorticity 
c   for stokes and stresses for elasticity
c
c   1) Analytic gradient (grada) 
c      =  \frac{\partial \phi(z_i)}{\partial z}
c
c   2) Anti analytic gradient (gradaa)
c       = z \frac{\partial^2 \phi_(z_i)} {\partial^2 z}_bar +
c         \frac{\partial \psi_(z_i)} {\partial z}
c 
c   The main FMM routine permits both evaluation at sources
c   and at a collection of targets.
c 
c   The main FMM routine permits both evaluation at sources
c   and at a collection of targets. 
c   This subroutine is used to simplify the user interface 
c   (by setting the number of targets to zero) and calling the more 
c   general FMM.
c
c   See below for explanation of calling sequence arguments.
c  
        real *8 source(2,*)
        complex *16 charge(*)
        complex *16 dip1(*)
        complex *16 dip2(*)
        complex *16 ima
        complex *16 vel(*)
        complex *16 grada(*)
        complex *16 gradaa(*)
c
        real *8 target(2,1)
        complex *16 veltarg
        complex *16 gradatarg
        complex *16 gradaatarg        
c
        data ima/(0.0d0,1.0d0)/
c       
        ntarget=0
        ifveltarg=0
        ifgradatarg=0
        ifgradaatarg=0
c
        call bhfmm2dparttarg(ier,iprec,nsource,source,
     $  ifcharge,charge,ifdipole,dip1,dip2,
     $  ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $  ntarget,target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $  ifgradaatarg,gradaatarg)
c
        return
        end
c
c***********************************************************************
        subroutine bhfmm2dparttarg(ier,iprec,nsource,source,
     $  ifcharge,charge,ifdipole,dip1,dip2,
     $  ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $  ntarget,target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $  ifgradaatarg,gradaatarg)
        implicit real *8 (a-h,o-z)
c       
c   Generalized biharmonic FMM in R^2: evaluate all pairwise particle
c   interactions (ignoring self-interaction) 
c   and interactions with targets.
c
c   Self-interactions are not included.
c   
c   bhfmm2d: charge and dipstr are complex valued, z are complex numbers.
c
c   bhfmm2d: charge and dip1,dip2 are complex valued,
c   z are complex numbers. The complex velocity for stokes/
c   displacement for stokes/elasticity problem take the general form 
c
c   Stokes:
c   vel(z_i) = \sum_{j \neq i} charge_j *\log|z_i-z_j| + 
c                (charge_j)_bar (z_i-z_j)/(z_i-z_j)_bar+
c                dip1/(z_i-z_j) + dip2/(z_i-z_j)_bar-
c                dip1_bar (z_i-z_j)/(z_i-z_j)^2_bar
c
c   The velocity for stokes is related to the goursat functions
c   as vel = \phi(z) + z (d/dz (\phi))_bar+\psi(z)
c 
c   The goursat functions are given by
c
c   \phi(z_i) = \sum_{j\neq i} charge_j log(z_i-z_j)+
c                     dip1/(z_i-z_j)
c
c   \psi(z_i) = \sum_{j\neq i} dippar2_bar/(z_i-z_j)+
c               z_j_bar dippar1/(z_i-z_j)^2 +
c               charge_j_bar log(z_i-z_j)- z_j_bar charge_j/(z_i-z_j)
c
c   For stokes and elasticity problems the following derivatives are
c   relevant to physical quantities such as pressure and vorticity 
c   for stokes and stresses for elasticity
c
c   1) Analytic gradient (grada) 
c      =  \frac{\partial \phi(z_i)}{\partial z}
c
c   2) Anti analytic gradient (gradaa)
c       = z \frac{\partial^2 \phi_(z_i)} {\partial^2 z}_bar +
c         \frac{\partial \psi_(z_i)} {\partial z}
c 
c   The main FMM routine permits both evaluation at sources
c   and at a collection of targets.
c 
c
c   This is primarily a memory management code. 
c   The actual work is carried out in subroutine cfmm2dparttargmain.
c-----------------------------------------------------------------------
c   INPUT PARAMETERS:
c
c   iprec:  FMM precision flag
c
c                 -2 => tolerance =.5d0
c                 -1 => tolerance =.5d-1
c                  0 => tolerance =.5d-2
c                  1 => tolerance =.5d-3
c                  2 => tolerance =.5d-6
c                  3 => tolerance =.5d-9
c                  4 => tolerance =.5d-12
c                  5 => tolerance =.5d-15
c
c       nsource: integer:  number of sources
c       source: real *8 (2,nsource):  source locations

c   ifcharge:  charge computation flag
c              ifcharge = 1   =>  include charge contribution
c                                     otherwise do not
c   charge: complex *16 (nsource): charge strengths

c   ifdipole:  dipole computation flag
c              ifdipole = 1   =>  include dipole contribution
c                                     otherwise do not
c   dip1: complex *16 (nsource): dip1 strengths
c   dip2: complex *16 (nsource): dip2 strengths. 
c
c   ifvel:  velocity flag (1=compute velocity, otherwise no)
c   ifgrada: Analytic gradient flag 
c            (1=compute analytic gradient, otherwise no)
c   ifgradaa: Anti analytic gradient flag 
c               (1=compute anti analytic gradient, otherwise no)

c   ntarget: integer:  number of targets
c   target: real *8 (2,ntarget):  target locations

c   ifveltarg:  target velocity flag 
c             (1=compute velocity, otherwise no)
c   ifgradtarg:  target analytic gradient flag 
c             (1=compute analytic gradient, otherwise no)
c   ifgradaatarg:  target anti analytic flag 
c             (1=compute anti analytic, otherwise no)
c--------------------------------------------------------------------
c   OUTPUT PARAMETERS:
c
c   ier   =  error return code
c                ier=0     =>  normal execution
c                ier=4     =>  cannot allocate tree workspace
c                ier=8     =>  cannot allocate bulk FMM  workspace
c                ier=16    =>  cannot allocate mpole expansion
c                              workspace in FMM
c
c   vel: complex *16 (nsource): velocity at source locations
c   grada: complex *16 (nsource): analytic gradient at source locations
c   gradaa: complex *16 (nsource): 
c                         anti analytic gradient at source locations
c   veltarg: complex *16 (ntarget): velocity at target locations 
c   gradatarg: complex *16 (ntarget): 
c                         analytic gradient  at target locations 
c   gradaatarg: complex *16 (ntarget): 
c                         anti analytic gradient at target locations
c
        real *8 source(2,*)
        complex *16 charge(*)
        complex *16 dip1(*)
        complex *16 dip2(*)
        complex *16 ima
        complex *16 vel(*)
        complex *16 grada(*)
        complex *16 gradaa(*)
c
        real *8 target(2,*)
        complex *16 veltarg(*)
        complex *16 gradatarg(*)        
        complex *16 gradaatarg(*)
c
        real *8 timeinfo(10)
c       
        real *8 center(2)
c       
c
c     Note: various arrays dimensioned here to 200.
c     That allows for 200 levels of refinement, which is 
c     more than enough for any non-pathological case.
c
        integer laddr(2,200)
        real *8 scale(0:200)
        real *8 bsize(0:200)
        integer nterms(0:200)
c       
        complex *16 vtemp,gatemp,gaatemp
c       
        integer box(15)
        real *8 center0(2),corners0(2,4)
c       
        integer box1(15)
        real *8 center1(2),corners1(2,4)
c       
        real *8, allocatable :: w(:)
        real *8, allocatable :: wlists(:)
        real *8, allocatable :: wrmlexp(:)
c
        data ima/(0.0d0,1.0d0)/
c       
        ier=0
        lused7=0
c       
        done=1
        pi=4*atan(done)
c
c
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c       
        ifprint=0
c
c     set fmm tolerance based on iprec flag.
c       
        if( iprec .eq. -2 ) epsfmm=.5d-0 
        if( iprec .eq. -1 ) epsfmm=.5d-1
        if( iprec .eq. 0 ) epsfmm=.5d-2
        if( iprec .eq. 1 ) epsfmm=.5d-3
        if( iprec .eq. 2 ) epsfmm=.5d-6
        if( iprec .eq. 3 ) epsfmm=.5d-9
        if( iprec .eq. 4 ) epsfmm=.5d-12
        if( iprec .eq. 5 ) epsfmm=.5d-15
        if( iprec .eq. 6 ) epsfmm=0
c       
        if(ifprint .eq. 1) call prin2('epsfmm=*',epsfmm,1)
c
c     set criterion for box subdivision (number of sources per box)
c
        if( iprec .eq. -2 ) nbox=3
        if( iprec .eq. -1 ) nbox=5
        if( iprec .eq. 0 ) nbox=8
        if( iprec .eq. 1 ) nbox=10
        if( iprec .eq. 2 ) nbox=15
        if( iprec .eq. 3 ) nbox=20
        if( iprec .eq. 4 ) nbox=25
        if( iprec .eq. 5 ) nbox=45
        if( iprec .eq. 6 ) nbox=nsource+ntarget
c
        if(ifprint .eq. 1) call prinf('nbox=*',nbox,1)
c
c       create quad-tree data structure
        t1=second()
C$      t1=omp_get_wtime() 
        ntot = 100*(nsource+ntarget)+10000
        do ii = 1,10
           allocate (wlists(ntot))
           call bhfmm2dparttree(ier,iprec,
     $     nsource,source,ntarget,target,
     $     nbox,epsfmm,iisource,iitarget,iwlists,lwlists,
     $     nboxes,laddr,nlev,center,size,
     $     wlists,ntot,lused7)
           if (ier.ne.0) then
              deallocate(wlists)
              ntot = ntot*1.5
              call prinf(' increasing allocation, ntot is *',ntot,1)
           else
              goto 1200
           endif
        enddo
1200    continue

        if (ier.ne.0) then
           call prinf(' exceeded max allocation, ntot is *',ntot,1)
           ier = 4
           return
        endif
        t2=second()
C$      t2=omp_get_wtime() 
        if( ifprint .eq. 1 ) call prin2('time in d2tstcr=*',t2-t1,1)
c
c     lused7 is counter that steps through workspace,
c     keeping track of total memory used.
c
        lused7=1
c
c       ... prepare data structures 
c
        do i = 0,nlev
           scale(i) = 1.0d0
           boxsize = abs((size/2.0**i))
           scale(i) = boxsize
        enddo
c       
        if(ifprint .eq. 1) call prin2('scale=*',scale,nlev+1)
c       
c       carve up workspace further
c
c     isourcesort is pointer for sorted source coordinates
c     itargetsort is pointer for sorted target locations
c     ichargesort is pointer for sorted charge densities
c     idip1sort is pointer for sorted dipole strength 1
c     idip2sort is pointer for sorted dipole strength 2
c
        isourcesort = lused7 + 5
        lsourcesort = 2*nsource
        itargetsort = isourcesort+lsourcesort
        ltargetsort = 2*ntarget
        ichargesort = itargetsort+ltargetsort
        lchargesort = 2*nsource
        idip1sort = ichargesort+lchargesort
        if (ifdipole.eq.1) then
          ldip1 = 2*nsource
          ldip2 = 2*nsource
        else
          ldip1 = 2
          ldip2 = 2
        endif
        idip2sort = idip1sort + ldip1
        lused7 = idip2sort + ldip2
c
c       ... allocate the potential and gradient arrays
c
        ivel = lused7
        lvel = 2*nsource
        lused7=lused7+lvel
c       
        igrada = lused7
        if(ifgrada.eq. 1) then
            lgrada = 2*nsource
        else
            lgrada=2
        endif
        lused7=lused7+lgrada
c      
        igradaa = lused7
        if(ifgradaa .eq. 1) then
            lgradaa=2*nsource
        else
            lgradaa=2
        endif
        lused7=lused7+lgradaa
c      
        iveltarg = lused7
        lveltarg = 2*ntarget
        lused7=lused7+lveltarg
c       
        igradatarg = lused7
        if( ifgradatarg .eq. 1) then
            lgradatarg=2*ntarget
        else
            lgradatarg=2
        endif
        lused7=lused7+lgradatarg
c      
        igradaatarg = lused7
        if(ifgradaatarg .eq. 1) then
            lgradaatarg = 2*ntarget
        else
            lgradaatarg=2
        endif
        lused7=lused7+lgradaatarg
c      
        if(ifprint .eq. 1) call prinf(' lused7 is *',lused7,1)
c
c       based on FMM tolerance, compute expansion lengths nterms(i)
c      
        nmax = 0

        do i = 0,nlev
           bsize(i)=size/2.0d0**i
           call l2dterms(epsfmm, nterms(i), ier)
           if (nterms(i).gt. nmax .and. i.ge. 2) nmax = nterms(i)
        enddo
c
        if (ifprint.eq.1) 
     $     call prin2('in bhfmm2dpart, bsize(0)=*',
     $     abs(bsize(0)),1)
c
        if (ifprint.eq.1) call prin2('bsize=*',bsize,nlev+1)
        if (ifprint.eq.1) call prinf('nterms=*',nterms,nlev+1)
c
c       
c     Multipole and local expansions will be held in workspace
c     in locations pointed to by array iaddr(10,nboxes).
c
c     iiaddr is pointer to iaddr array, itself contained in workspace.
c     imptemp is pointer for single expansion (dimensioned by nmax)
c
c       ... allocate iaddr and temporary arrays
c
        iiaddr = lused7 
        imptemp1 = iiaddr + 10*nboxes
        lmptemp1 = (2*nmax+1)*2
        lused7 = imptemp1 + lmptemp1
   
        imptemp2 = lused7
        lmptemp2 = (2*nmax+1)*2
        lused7 = imptemp2 + lmptemp2

        imptemp3 = lused7
        lmptemp3 = (2*nmax+1)*2
        lused7 = imptemp3 + lmptemp3

        imptemp4 = lused7
        lmptemp4 = (2*nmax+1)*2
        lused7 = imptemp4 + lmptemp4

        imptemp5 = lused7
        lmptemp5 = (2*nmax+1)*2
        lused7 = imptemp5 + lmptemp5

        allocate(w(lused7),stat=ier)
        if (ier.ne.0) then
           call prinf(' cannot allocate bulk FMM workspace,
     1                  lused7 is *',lused7,1)
           ier = 8
           return
        endif
c
c     reorder sources, targets so that each box holds
c     contiguous list of source/target numbers.

c
        call bh2dreorder(nsource,source,ifcharge,charge,
     $  wlists(iisource),ifdipole,dip1,dip2,w(isourcesort),
     1  w(ichargesort),w(idip1sort),w(idip2sort)) 
c       
        call bh2dreordertarg(ntarget,target,wlists(iitarget),
     $  w(itargetsort))
c
        if(ifprint .eq. 1) then
           call prinf('finished reordering=*',ier,1)
           call prinf('ier=*',ier,1)
           call prinf('nboxes=*',nboxes,1)
           call prinf('nlev=*',nlev,1)
           call prinf('nboxes=*',nboxes,1)
           call prinf('lused7=*',lused7,1)
        endif
c
c     allocate memory need by multipole, local expansions at all
c     levels
c     irmlexp is pointer for workspace need by various fmm routines,
c
        call bh2dmpalloc(wlists(iwlists),w(iiaddr),nboxes,lmptot,nterms)
        if(ifprint .eq. 1) call prinf(' lmptot is *',lmptot,1)
c       
        irmlexp = 1
        lused7 = irmlexp + lmptot 
        if(ifprint .eq. 1) call prinf(' lused7 is *',lused7,1)
        allocate(wrmlexp(lused7),stat=ier)
        if (ier.ne.0) then
           call prinf(' cannot allocate mpole expansion workspace,
     1     lused7 is *',lused7,1)
           ier = 16
           return
        endif
c
c     Memory allocation is complete. 
c     Call main fmm routine. There are, unfortunately, a lot
c     of parameters here. ifevalfar and ifevalloc determine
c     whether far gradient and local gradients (respectively) are to 
c     be evaluated. Setting both to 1 means that both will be
c     computed (which is the normal scenario).
c
        ifevalfar=1
        ifevalloc=1
c
        t1=second()
C$      t1=omp_get_wtime() 
        call bhfmm2dparttargmain(ier,iprec,
     $  ifevalfar,ifevalloc,
     $  nsource,w(isourcesort),wlists(iisource),
     $  ifcharge,w(ichargesort),
     $  ifdipole,w(idip1sort),w(idip2sort),
     $  ifvel,w(ivel),ifgrada,w(igrada),ifgradaa,w(igradaa),
     $  ntarget,w(itargetsort),wlists(iitarget),
     $  ifveltarg,w(iveltarg),ifgradatarg,w(igradatarg),
     $  ifgradaatarg,w(igradaatarg),
     $  epsfmm,w(iiaddr),wrmlexp(irmlexp),w(imptemp1),lmptemp1,
     $  w(imptemp2),lmptemp2,w(imptemp3),lmptemp3,w(imptemp4),
     $  lmptemp4,w(imptemp5),lmptemp5,
     $  nboxes,laddr,nlev,scale,bsize,nterms,
     $  wlists(iwlists),lwlists)
        t2=second()
C$      t2=omp_get_wtime() 
        if( ifprint .eq. 1 ) call prin2('time in fmm main=*',t2-t1,1)
c
c       parameter ier from targmain routine is currently 
c       meaningless, reset to 0
        if( ier .ne. 0 ) ier = 0
c
        if(ifprint .eq. 1) call prinf('lwlists=*',lwlists,1)
        if(ifprint .eq. 1) then
           call prinf('lused total=*',lused7,1)
           call prinf('lused total(k)=*',lused7/1000,1)
           call prinf('lused total(M)=*',lused7/1000000,1)
        endif
c       
        if(ifprint .eq. 1) 
     $     call prin2('memory / point = *',(lused7)/dble(nsource),1)
c       
c
        if(ifvel .eq. 1) 
     $     call bh2dpsort(nsource,wlists(iisource),w(ivel),vel)
        if(ifgrada .eq. 1) 
     $     call bh2dpsort(nsource,wlists(iisource),w(igrada),grada)
        if(ifgradaa .eq. 1) 
     $     call bh2dpsort(nsource,wlists(iisource),w(igradaa),gradaa)
c
        if(ifveltarg .eq. 1 )
     $     call bh2dpsort(ntarget,wlists(iitarget),w(iveltarg),veltarg)
        if(ifgradatarg .eq. 1) 
     $     call bh2dpsort(ntarget,wlists(iitarget),w(igradatarg),
     $     gradatarg)
        if(ifgradaatarg .eq. 1) 
     $   call bh2dpsort(ntarget,wlists(iitarget),w(igradaatarg),
     $   gradaatarg)
c       
        return
        end
c
c***********************************************************************
        subroutine bhfmm2dparttargmain(ier,iprec,
     $  ifevalfar,ifevalloc,
     $  nsource,sourcesort,isource,
     $  ifcharge,chargesort,
     $  ifdipole,dip1sort,dip2sort,
     $  ifvel,vel,ifgrada,grada,ifgradaa,gradaa,ntarget,
     $  targetsort,itarget,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $  ifgradaatarg,gradaatarg,
     $  epsfmm,iaddr,rmlexp,mptemp1,lmptemp1,mptemp2,lmptemp2,
     $  mptemp3,lmptemp3,mptemp4,lmptemp4,mptemp5,lmptemp5,
     $  nboxes,laddr,nlev,scale,bsize,nterms,
     $  wlists,lwlists)
        implicit real *8 (a-h,o-z)
        real *8 sourcesort(2,*)
        integer isource(*)
        complex *16 chargesort(*)
        complex *16 dip1sort(*)
        complex *16 dip2sort(*)
        complex *16 ima
        complex *16 vel(*)
        complex *16 grada(*)
        complex *16 gradaa(*)
        real *8 targetsort(2,*)
        integer itarget(*)
        complex *16 veltarg(*)
        complex *16 gradatarg(*)
        complex *16 gradaatarg(*)
        real *8 wlists(*)
        integer iaddr(10,nboxes)
        real *8 rmlexp(*)
        complex *16 mptemp1(lmptemp1)
        complex *16 mptemp2(lmptemp2)
        complex *16 mptemp3(lmptemp3)
        complex *16 mptemp4(lmptemp4)
        complex *16 mptemp5(lmptemp5)
        real *8 timeinfo(10)
        real *8 center(3)
        integer laddr(2,200)
        real *8 scale(0:200)
        real *8 bsize(0:200)
        integer nterms(0:200)
        integer list(10 000)
        complex *16 vtemp,gatemp,gaatemp
        integer box(15)
        real *8 center0(2),corners0(2,4)
        integer box1(15)
        real *8 center1(2),corners1(2,4)
        integer nterms_eval(4,0:200)
c
        data ima/(0.0d0,1.0d0)/
c
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c     Prints timing breakdown, list information, and other things if ifprint=2.
c       
        ifprint=0
c
c
c       ... set the potential and gradient to zero
c
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
        do i=1,nsource
           if( ifvel .eq. 1) vel(i)=0
           if( ifgrada .eq. 1) grada(i)=0
           if( ifgradaa .eq. 1) gradaa(i)=0
        enddo
C$OMP END PARALLEL DO
c       
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
        do i=1,ntarget
           if( ifveltarg .eq. 1) veltarg(i)=0
           if( ifgradatarg .eq. 1) then
              gradatarg(i)=0
           endif
           if( ifgradaatarg .eq. 1) then
              gradaatarg(i)=0
           endif
        enddo
C$OMP END PARALLEL DO
c
        do i=1,10
           timeinfo(i)=0
        enddo
c
c
        if( ifevalfar .eq. 0 ) goto 8000
c
        if (ifprint .ge. 2) then 
           call prinf('nterms_eval=*',nterms_eval,4*(nlev+1))
        endif
c
c       ... set all multipole and local expansions to zero
c
C$OMP PARALLEL DO DEFAULT(SHARED) 
C$OMP$PRIVATE(ibox,box,center0,corners0,level,ier)
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do ibox = 1,nboxes
           call d2tgetb(ier,ibox,box,center0,corners0,wlists)
           level=box(1)
           if( level .ge. 2 ) then
             call bh2dzero(rmlexp(iaddr(1,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(2,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(3,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(4,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(5,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(6,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(7,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(8,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(9,ibox)),nterms(level))
             call bh2dzero(rmlexp(iaddr(10,ibox)),nterms(level))
           endif
        enddo
C$OMP END PARALLEL DO
c
c
        if(ifprint .ge. 1) then
           call prinf('=== STEP 1 (form mp) ====*',i,0)
        endif
        t1=second()
C$      t1=omp_get_wtime() 
c
c       ... step 1, locate all charges, assign them to boxes, and
c       form multipole expansions
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,radius)
C$OMP$PRIVATE(mptemp1,lused,ier,i,j,vtemp,gatemp,gaatemp,cd)
C$OMP$PRIVATE(mptemp2,mptemp3,mptemp4,mptemp5) 
C$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1) 
        do 1200 ibox=1,nboxes
c
           call d2tgetb(ier,ibox,box,center0,corners0,wlists)
           call d2tnkids(box,nkids)
c
           level=box(1)
           if( level .lt. 2 ) goto 1200
c
c
           if (ifprint .ge. 2) then
              call prinf('ibox=*',ibox,1)
              call prinf('box=*',box,15)
              call prinf('nkids=*',nkids,1)
           endif
c
           if (nkids .eq. 0) then
              npts=box(10)
              if (ifprint .ge. 2) then
                 call prinf('npts=*',npts,1)
                 call prinf('isource=*',isource(box(9)),box(10))
              endif
           endif
c
c       ... prune all sourceless boxes
c
c       ... form multipole expansions
           if( box(10) .eq. 0 ) goto 1200
           if (nkids .eq. 0) then
               radius = (corners0(1,1) - center0(1))**2
               radius = radius + (corners0(2,1) - center0(2))**2
               radius = sqrt(radius)
c
               call bh2dzero(rmlexp(iaddr(1,ibox)),nterms(level))
               call bh2dzero(rmlexp(iaddr(2,ibox)),nterms(level))
               call bh2dzero(rmlexp(iaddr(3,ibox)),nterms(level))
               call bh2dzero(rmlexp(iaddr(4,ibox)),nterms(level))
               call bh2dzero(rmlexp(iaddr(5,ibox)),nterms(level))
               if_use_trunc = 0

               call bh2dformmp(ier,scale(level),
     1         sourcesort(1,box(9)),ifcharge,chargesort(box(9)),
     2         ifdipole,dip1sort(box(9)),dip2sort(box(9)),npts,
     3         center0,nterms(level),rmlexp(iaddr(1,ibox)),
     4         rmlexp(iaddr(2,ibox)),rmlexp(iaddr(3,ibox)),
     5         rmlexp(iaddr(4,ibox)),rmlexp(iaddr(5,ibox)))

             endif
 1200     continue
C$OMP END PARALLEL DO
c
          t2=second()
C$      t2=omp_get_wtime() 
          timeinfo(1)=t2-t1
c       
          if(ifprint .ge. 1) then
              call prinf('=== STEP 2 (form lo) ====*',i,0)
          endif
          t1=second()
C$      t1=omp_get_wtime() 
c       ... step 2, adaptive part, form local expansions, 
c           or evaluate the potentials and gradients directly
c 
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level0,itype,list,nlist)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1,ifdirect3,radius)
C$OMP$PRIVATE(lused,ier,i,j,vtemp,gatemp,gaatemp,cd,ilist,npts) 
C$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1) 
          do 3251 ibox=1,nboxes
             call d2tgetb(ier,ibox,box,center0,corners0,wlists)
c
             itype=4
             call d2tgetl(ier,ibox,itype,list,nlist,wlists)
             if (nlist .gt. 0) then 
                if (ifprint .ge. 2) then
                    call prinf('ibox=*',ibox,1)
                    call prinf('list3=*',list,nlist)
                endif
             endif
c       ... note that lists 3 and 4 are dual
c
c       ... form local expansions for all boxes in list 3
c       ... if target is childless, evaluate directly (if cheaper)
c        
             do 3250 ilist=1,nlist
                 jbox=list(ilist)
                 call d2tgetb(ier,jbox,box1,center1,corners1,wlists)
c        
                 npts=box1(10)            
                 if( npts .eq. 0 ) goto 3250
c
                 level0=box(1)
                 level1=box1(1)
c
                 ifdirect3 = 0
c
                 if( ifdirect3 .eq. 0 ) then
                     npts=box1(10)
                     if_use_trunc = 0

                     call bh2dformta_add(ier,scale(level0),
     $               sourcesort(1,box1(9)),ifcharge,
     1               chargesort(box1(9)),ifdipole,dip1sort(box1(9)),
     2               dip2sort(box1(9)),npts,center0,nterms(level0),
     3               rmlexp(iaddr(6,ibox)),rmlexp(iaddr(7,ibox)),
     4               rmlexp(iaddr(8,ibox)),rmlexp(iaddr(9,ibox)),
     5               rmlexp(iaddr(10,ibox)))
                 else
                    call bhfmm2dpart_direct(box1,box,sourcesort,
     $              ifcharge,chargesort,ifdipole,dip1sort,dip2sort,
     $              ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $              targetsort,ifveltarg,veltarg,ifgradatarg,
     $              gradatarg,ifgradaatarg,gradaatarg)
                 endif
 3250         continue
 3251     continue
C$OMP END PARALLEL DO
          t2=second()
C$      t2=omp_get_wtime() 
          timeinfo(2)=t2-t1
c
          if(ifprint .ge. 1)
     $         call prinf('=== STEPS 3,4,5 ====*',i,0)
          ifprune_list2 = 1
          if (ifvel.eq.1) ifprune_list2 = 0
          if (ifgrada.eq.1) ifprune_list2 = 0
          if (ifgradaa.eq.1) ifprune_list2 = 0
          call bhfmm2d_list2
     $    (bsize,nlev,laddr,scale,nterms,rmlexp,iaddr,epsfmm,
     $    timeinfo,wlists,mptemp1,lmptemp1,mptemp2,lmptemp2,
     $    mptemp3,lmptemp3,mptemp4,lmptemp4,
     $    mptemp5,lmptemp5,ifprune_list2)
c
          if(ifprint .ge. 1)
     $       call prinf('=== STEP 6 (eval mp) ====*',i,0)
          t1=second()
C$      t1=omp_get_wtime() 
c       ... step 6, adaptive part, evaluate multipole expansions, 
c           or evaluate the potentials and gradients directly
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,itype,list,nlist)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1,ifdirect4,radius)
C$OMP$PRIVATE(lused,ier,i,j,vtemp,gatemp,gaatemp,cd,ilist,level) 
C$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1) 
          do 3252 ibox=1,nboxes
             call d2tgetb(ier,ibox,box,center0,corners0,wlists)
             itype=3
             call d2tgetl(ier,ibox,itype,list,nlist,wlists)
             if (nlist .gt. 0) then 
                if (ifprint .ge. 2) then
                   call prinf('ibox=*',ibox,1)
                   call prinf('list4=*',list,nlist)
                endif
             endif
c       ... note that lists 3 and 4 are dual
c
c       ... evaluate multipole expansions for all boxes in list 4 
c       ... if source is childless, evaluate directly (if cheaper)
c
             do ilist=1,nlist
                jbox=list(ilist)
                call d2tgetb(ier,jbox,box1,center1,corners1,wlists)
                level=box1(1)
c
                ifdirect4 = 0
                if (ifdirect4 .eq. 0) then
                   if( box(10) .gt. 0) then
                      call bh2dmpevalall(scale(level),center1,
     1                rmlexp(iaddr(1,jbox)),rmlexp(iaddr(2,jbox)),
     2                rmlexp(iaddr(3,jbox)),rmlexp(iaddr(4,jbox)),
     3                rmlexp(iaddr(5,jbox)),nterms(level),
     4                sourcesort(1,box(9)),box(10),vel(box(9)),
     5                ifgrada,grada(box(9)),ifgradaa,gradaa(box(9)))
                   endif

                   if( box(12) .gt. 0 ) then
                      call bh2dmpevalall(scale(level),center1,
     1                rmlexp(iaddr(1,jbox)),rmlexp(iaddr(2,jbox)),
     2                rmlexp(iaddr(3,jbox)),rmlexp(iaddr(4,jbox)),
     3                rmlexp(iaddr(5,jbox)),nterms(level),
     4                targetsort(1,box(11)),box(12),veltarg(box(11)),
     5                ifgradatarg,gradatarg(box(11)),
     6                ifgradaatarg,gradaatarg(box(11)))
                   endif
                else
                   call bhfmm2dpart_direct(box1,box,sourcesort,
     $             ifcharge,chargesort,ifdipole,dip1sort,dip2sort,
     $             ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $             targetsort,ifveltarg,veltarg,ifgradatarg,
     $             gradatarg,ifgradaatarg,gradaatarg)
                endif
             enddo
 3252     continue
C$OMP END PARALLEL DO
          t2=second()
C$      t2=omp_get_wtime() 
          timeinfo(6)=t2-t1
          if(ifprint .ge. 1)
     $       call prinf('=== STEP 7 (eval lo) ====*',i,0)
          t1=second()
C$      t1=omp_get_wtime() 
c
c       ... step 7, evaluate local expansions
c       and all gradients directly
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,ier)
C$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1) 
          do 6201 ibox=1,nboxes
             call d2tgetb(ier,ibox,box,center0,corners0,wlists)
             call d2tnkids(box,nkids)
             if (ifprint .ge. 2) then
                call prinf('ibox=*',ibox,1)
                call prinf('box=*',box,15)
                call prinf('nkids=*',nkids,1)
             endif
c
             if (nkids .eq. 0) then
                npts=box(10)
                if (ifprint .ge. 2) then
                   call prinf('npts=*',npts,1)
                   call prinf('isource=*',isource(box(9)),box(10))
                endif
             endif
             if (nkids .eq. 0) then
c       ... evaluate local expansions
                level=box(1)
                npts=box(10)
                if (level .ge. 2) then
                   if( box(10) .gt. 0 ) then

                      call bh2dtaevalall(scale(level),center0,
     1                rmlexp(iaddr(6,ibox)),rmlexp(iaddr(7,ibox)),
     2                rmlexp(iaddr(8,ibox)),rmlexp(iaddr(9,ibox)),
     3                rmlexp(iaddr(10,ibox)),nterms(level),
     4                sourcesort(1,box(9)),box(10),vel(box(9)),
     5                ifgrada,grada(box(9)),ifgradaa,gradaa(box(9)))
                   endif
                   if( box(12) .gt. 0 ) then
                      call bh2dtaevalall(scale(level),center0,
     1                rmlexp(iaddr(6,ibox)),rmlexp(iaddr(7,ibox)),
     2                rmlexp(iaddr(8,ibox)),rmlexp(iaddr(9,ibox)),
     3                rmlexp(iaddr(10,ibox)),nterms(level),
     4                targetsort(1,box(11)),box(12),veltarg(box(11)),
     5                ifgradatarg,gradatarg(box(11)),
     6                ifgradaatarg,gradaatarg(box(11)))
                   endif
                endif
             endif
 6201     continue
C$OMP END PARALLEL DO
          t2=second()
C$      t2=omp_get_wtime() 
          timeinfo(7)=t2-t1
 8000     continue

          if( ifevalloc .eq. 0 ) goto 9000
             if(ifprint .ge. 1)
     $          call prinf('=== STEP 8 (direct) =====*',i,0)
             t1=second()
C$      t1=omp_get_wtime() 
c
c       ... step 8, evaluate direct interactions 
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,nkids,list,nlist,npts)
C$OMP$PRIVATE(jbox,box1,center1,corners1)
C$OMP$PRIVATE(ier,ilist,itype) 
C$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1)
             do 6202 ibox=1,nboxes
                call d2tgetb(ier,ibox,box,center0,corners0,wlists)
                call d2tnkids(box,nkids)
                if (ifprint .ge. 2) then
                   call prinf('ibox=*',ibox,1)
                   call prinf('box=*',box,15)
                   call prinf('nkids=*',nkids,1)
                endif
                if (nkids .eq. 0) then
                   npts=box(10)
                   if (ifprint .ge. 2) then
                      call prinf('npts=*',npts,1)
                      call prinf('isource=*',isource(box(9)),box(10))
                   endif
                endif

                if (nkids .eq. 0) then
c       ... evaluate self interactions
                   call bhfmm2dpart_direct_self(box,sourcesort,
     $             ifcharge,chargesort,ifdipole,dip1sort,dip2sort,
     $             ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $             targetsort,ifveltarg,veltarg,ifgradatarg,
     $             gradatarg,ifgradaatarg,gradaatarg)
c       ... retrieve list #1
c       ... evaluate interactions with the nearest neighbours
c
                   itype=1
                   call d2tgetl(ier,ibox,itype,list,nlist,wlists)
                   if (ifprint.ge.2) call prinf('list1=*',list,nlist)
c
c       ... for all pairs in list #1, 
c       evaluate the potentials and gradients directly
c
                   do 6203 ilist=1,nlist
                      jbox=list(ilist)
                      call d2tgetb(ier,jbox,box1,center1,corners1,
     $                wlists)
c
c       ... prune all sourceless boxes
c
                      if( box1(10) .eq. 0 ) goto 6203
                      call bhfmm2dpart_direct(box1,box,sourcesort,
     $                ifcharge,chargesort,ifdipole,dip1sort,dip2sort,
     $                ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $                targetsort,ifveltarg,veltarg,ifgradatarg,
     $                gradatarg,ifgradaatarg,gradaatarg)
 6203              continue
                endif
 6202        continue
C$OMP END PARALLEL DO
c
          t2=second()
C$      t2=omp_get_wtime() 
          timeinfo(8)=t2-t1
 9000     continue
          if (ifprint .ge. 1) call prin2('timeinfo=*',timeinfo,8)
          d=0
          do i=1,8
             d=d+timeinfo(i)
          enddo
c       
          if (ifprint .ge. 1) call prin2('sum(timeinfo)=*',d,1)
c
          if (ifprint .ge. 1) then
             call prinf('nboxes=*',nboxes,1)
             open(unit=25,file='output.dat',access='append')
             write(25,*) "nboxes=",nboxes
             close(25)
             call prinf('nsource=*',nsource,1)
             call prinf('ntarget=*',ntarget,1)
          endif
       
        return
        end
c
c
c
c
c
        subroutine bhfmm2dpart_direct_self(box,
     $     source,ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $     target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $     ifgradaatarg,gradaatarg)
        implicit real *8 (a-h,o-z)
c
        integer box(15),box1(15)
c
        real *8 source(2,*)
        complex *16 charge(*),dip1(*),dip2(*)
        real *8 target(2,*)
c
        complex *16 vel(*),grada(*),gradaa(*)
        complex *16 veltarg(*),gradatarg(*),gradaatarg(*)
        complex *16 vtemp,gatemp,gaatemp
c
c
        do 6160 j=box(9),box(9)+box(10)-1
        do 6150 i=box(9),box(9)+box(10)-1
            if (i .eq. j) goto 6150
            call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     1      ifdipole,dip1(i),dip2(i),source(1,j),vtemp,ifgrada,
     2      gatemp,ifgradaa,gaatemp)
            if (ifvel.eq. 1) vel(j)=vel(j)+vtemp
            if (ifgrada.eq. 1) then
               grada(j)=grada(j)+gatemp
            endif
            if (ifgradaa.eq. 1) then
               gradaa(j)=gradaa(j)+gaatemp
            endif
 6150   continue
 6160   continue
        do j=box(11),box(11)+box(12)-1
        do i=box(9),box(9)+box(10)-1
            call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     $      ifdipole,dip1(i),dip2(i),target(1,j),vtemp,
     1      ifgradatarg,gatemp,ifgradaatarg,gaatemp)
            if (ifveltarg .eq. 1) veltarg(j)=veltarg(j)+vtemp
            if (ifgradatarg .eq. 1) then
               gradatarg(j)=gradatarg(j)+gatemp
            endif
            if (ifgradaatarg .eq. 1) then
               gradaatarg(j)=gradaatarg(j)+gaatemp
            endif
        enddo
        enddo
c
        return
        end
c
c
c********************************************************************
        subroutine bhfmm2dpart_direct(box,box1,
     $     source,ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $     target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $     ifgradaatarg,gradaatarg)
        implicit real *8 (a-h,o-z)
c
        integer box(15),box1(15)
c
        real *8 source(2,*)
        complex *16 charge(*),dip1(*),dip2(*)
        real *8 target(2,*)
c
        complex *16 vel(*),grada(*),gradaa(*)
        complex *16 veltarg(*),gradatarg(*),gradaatarg(*)
        complex *16 vtemp,gatemp,gaatemp
c
c
        do j=box1(9),box1(9)+box1(10)-1
           call bh2ddireval(source(1,box(9)),box(10),ifcharge,
     1     charge(box(9)),ifdipole,dip1(box(9)),dip2(box(9)),
     2     source(1,j),vtemp,ifgrada,gatemp,ifgradaa,gaatemp)
           if (ifvel.eq. 1) vel(j)=vel(j)+vtemp
           if (ifgrada.eq. 1) then
             grada(j)=grada(j)+gatemp
           endif
           if (ifgradaa.eq. 1) then
             gradaa(j)=gradaa(j)+gaatemp
           endif
        enddo
c
        do j=box1(11),box1(11)+box1(12)-1
           call bh2ddireval(source(1,box(9)),box(10),ifcharge,
     1     charge(box(9)),ifdipole,dip1(box(9)),dip2(box(9)),
     2     target(1,j),vtemp,ifgradatarg,gatemp,ifgradaatarg,gaatemp)

           if (ifveltarg .eq. 1) veltarg(j)=veltarg(j)+vtemp
           if (ifgradatarg .eq. 1) then
             gradatarg(j)=gradatarg(j)+gatemp
           endif
           if (ifgradaatarg .eq. 1) then
             gradaatarg(j)=gradaatarg(j)+gaatemp
           endif
        enddo
c       
        return
        end
c
c
c
c
c
        subroutine bh2dpartdirect(nsource,
     $     source,ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $     ntarget,target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $     ifgradaatarg,gradaatarg)
        implicit real *8 (a-h,o-z)
c
c       Generalized Cauchy interactions in R^2: evaluate all pairwise particle
c       interactions (ignoring self-interaction) 
c       and interactions with targets via direct O(N^2) algorithm.
c
c       We use log(z) for the Green's function.
c       Self-interactions are not included.
c   
c c2d: charge and dipstr are complex valued, z are complex numbers.
c
c        Note, that the complex valued logarithm is a multi-valued
c        function, so the potential values have to be interpreted
c        carefully, if charges are specified.  For example, only the
c        real part of potential is meaningful for real valued charges.
c        The gradients and hessians are valid for arbitrary complex charges.
c
c \phi(z_i) = \sum_{j\ne i} charge_j \log(z_i-z_j) + dipstr_j \frac{1}{z_i-z_j}
c
c        In this routine, we define the gradient as the first
c        derivative with respect to z, and the hessian as the second
c        derivative with respect to z.
c
c \gradient \phi(z_i) = \frac{\partial \phi(z_i)}{\partial z}
c \hessian  \phi(z_i) = \frac{\partial^2 \phi(z_i)}{\partial z^2}
c
c       INPUT PARAMETERS:
c
c       nsource: integer:  number of sources
c       source: real *8 (2,nsource):  source locations
c       ifcharge:  charge computation flag
c                  ifcharge = 1   =>  include charge contribution
c                                     otherwise do not
c       charge: complex *16 (nsource): charge strengths
c       ifdipole:  dipole computation flag
c                  ifdipole = 1   =>  include dipole contribution
c                                     otherwise do not
c       dipstr: complex *16 (nsource): dipole strengths
c
c       ifpot:  potential flag (1=compute potential, otherwise no)
c       ifgrad:  gradient flag (1=compute gradient, otherwise no)
c       ifhess:  hessian flag (1=compute hessian, otherwise no)
c       ntarget: integer:  number of targets
c       target: real *8 (2,ntarget):  target locations
c       ifpottarg:  target potential flag 
c                   (1=compute potential, otherwise no)
c       ifgradtarg:  target gradient flag 
c                   (1=compute gradient, otherwise no)
c       ihesstarg:  target hessian flag 
c                   (1=compute hessian, otherwise no)
c
c       OUTPUT PARAMETERS:
c
c       pot: complex *16 (nsource): potential at source locations
c       grad: complex *16 (nsource): gradient  at source locations
c       hess: complex *16 (nsource): hessian at source locations
c       pottarg: complex *16 (ntarget): potential at target locations 
c       gradtarg: complex *16 (ntarget): gradient  at target locations 
c       hesstarg: complex *16 (ntarget): hessian at target locations
c
c
c
        real *8 source(2,*)
        complex *16 charge(*),dip1(*),dip2(*)
        real *8 target(2,*)
c
        complex *16 vel(*),grada(*),gradaa(*)
        complex *16 veltarg(*),gradatarg(*),gradaatarg(*)
        complex *16 vtemp,gatemp,gaatemp
c
c
        do i=1,nsource
        if( ifvel .eq. 1) vel(i)=0
        if( ifgrada.eq. 1) then
           grada(i)=0
        endif
        if( ifgradaa .eq. 1) then
           gradaa(i)=0
        endif
        enddo
c       
        do i=1,ntarget
        if( ifveltarg .eq. 1) veltarg(i)=0
        if( ifgradatarg .eq. 1) then
           gradatarg(i)=0
        endif
        if( ifgradaatarg .eq. 1) then
           gradaatarg(i)=0
        endif
        enddo
c
        if( ifvel.eq.1 .or. ifgrada.eq.1 .or. ifgradaa.eq.1) then
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,vtemp,gatemp,gaatemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do 6160 j=1,nsource
        do 6150 i=1,nsource
            if (i .eq. j) goto 6150
            call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     1      ifdipole,dip1(i),dip2(i),source(1,j),vtemp,ifgrada,
     2      gatemp,ifgradaa,gaatemp)
            if (ifvel.eq. 1) vel(j)=vel(j)+vtemp
            if (ifgrada.eq. 1) then
               grada(j)=grada(j)+gatemp
            endif
            if (ifgradaa.eq. 1) then
               gradaa(j)=gradaa(j)+gaatemp
            endif
 6150   continue
 6160   continue
C$OMP END PARALLEL DO
        endif
c
        if( ifveltarg .eq. 1 .or. ifgradatarg .eq. 1 
     $      .or. ifgradaatarg .eq. 1) then
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,vtemp,gatemp,gaatemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do j=1,ntarget
        do i=1,nsource
            call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     1      ifdipole,dip1(i),dip2(i),target(1,j),vtemp,ifgradatarg,
     2      gatemp,ifgradaatarg,gaatemp)
            if (ifveltarg .eq. 1) veltarg(j)=veltarg(j)+vtemp
            if (ifgradatarg .eq. 1) then
               gradatarg(j)=gradatarg(j)+gatemp
            endif
            if (ifgradaatarg .eq. 1) then
               gradaatarg(j)=gradaatarg(j)+gaatemp
            endif
        enddo
        enddo
C$OMP END PARALLEL DO
        endif
c
        return
        end
