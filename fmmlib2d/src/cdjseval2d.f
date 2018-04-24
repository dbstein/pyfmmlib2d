cc Copyright (C) 2009-2012: Leslie Greengard and Zydrunas Gimbutas
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a modified FreeBSD license
cc (see COPYING in home directory). 
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c    $Date: 2011-02-22 17:34:23 -0500 (Tue, 22 Feb 2011) $
c    $Revision: 1670 $
c
c
c     Computation of  Bessel functions via recurrence
c
c**********************************************************************
      subroutine jfuns2d(ier,nterms,z,scale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
      implicit real *8 (a-h,o-z)
c**********************************************************************
c
c PURPOSE:
c
c	This subroutine evaluates the first NTERMS  Bessel 
c	functions and if required, their derivatives.
c	It incorporates a scaling parameter SCALE so that
c       
c		fjs_n(z)=j_n(z)/SCALE^n
c		fjder_n(z)=\frac{\partial fjs_n(z)}{\partial z}
c
c	NOTE: The scaling parameter SCALE is meant to be used when
c             abs(z) < 1, in which case we recommend setting
c	      SCALE = abs(z). This prevents the fjs_n from 
c             underflowing too rapidly.
c	      Otherwise, set SCALE=1.
c	      Do not set SCALE = abs(z) if z could take on the 
c             value zero. 
c             In an FMM, when forming an expansion from a collection of
c             sources, set SCALE = min( abs(k*r), 1)
c             where k is the Helmholtz parameter and r is the box dimension
c             at the relevant level.
c
c INPUT:
c
c    nterms (integer): order of expansion of output array fjs 
c    z     (complex *16): argument of the  Bessel functions
c    scale    (real *8) : scaling factor (discussed above)
c    ifder  (integer): flag indicating whether to calculate "fjder"
c		          0	NO
c		          1	YES
c    lwfjs  (integer): upper limit of input arrays 
c                         fjs(0:1) and iscale(0:1)
c    iscale (integer): integer workspace used to keep track of 
c                         internal scaling
c
c OUTPUT:
c
c    ier    (integer): error return code 
c                         ier=0 normal return;
c                         ier=8 insufficient array dimension lwfjs
c    fjs   (complex *16): array of scaled Bessel functions.
c    fjder (complex *16): array of derivs of scaled Bessel functions.
c    ntop  (integer) : highest index in arrays fjs that is nonzero
c
c       NOTE, that fjs and fjder arrays must be at least (nterms+2)
c       complex *16 elements long.
c
c
      integer iscale(0:1)
      complex *16 wavek,fjs(0:1),fjder(0:1)
      complex *16 z,zinv,com,fj0,fj1,zscale,ztmp
c
      complex *16 psi,zmul,zsn,zmulinv
      complex *16 ima
      data ima/(0.0d0,1.0d0)/
c
      data upbound/1.0d+32/, upbound2/1.0d+40/, upbound2inv/1.0d-40/
      data tiny/1.0d-200/,done/1.0d0/,zero/0.0d0/
c
c ... Initializing ...
c
      ier=0
c
c       set to asymptotic values if argument is sufficiently small
c
      if (abs(z).lt.tiny) then
         fjs(0) = done
         do i = 1, nterms
            fjs(i) = zero
	 enddo
c
	 if (ifder.eq.1) then
	    do i=0,nterms
	       fjder(i)=zero
	    enddo
	    fjder(1)=done/(2*scale)
	 endif
c
         RETURN
      endif
c
c ... Step 1: recursion up to find ntop, starting from nterms
c
      ntop=0
      zinv=done/z
      fjs(nterms+1)=done
      fjs(nterms)=zero
c
      do 1200 i=nterms+1,lwfjs
         dcoef=2*i
         ztmp=dcoef*zinv*fjs(i)-fjs(i-1)
         fjs(i+1)=ztmp
c
         dd = dreal(ztmp)**2 + dimag(ztmp)**2
         if (dd .gt. upbound2) then
            ntop=i+1
            goto 1300
         endif
 1200 continue
 1300 continue
      if (ntop.le.2) then
         ier=8
         return
      endif
c
c ... Step 2: Recursion back down to generate the unscaled jfuns:
c             if magnitude exceeds UPBOUND2, rescale and continue the 
c	      recursion (saving the order at which rescaling occurred 
c	      in array iscale.
c
      do i=0,ntop
         iscale(i)=0
      enddo
c
      fjs(ntop)=zero
      fjs(ntop-1)=done
      do 2200 i=ntop-1,1,-1
	 dcoef=2*i
         ztmp=dcoef*zinv*fjs(i)-fjs(i+1)
         fjs(i-1)=ztmp
c
         dd = dreal(ztmp)**2 + dimag(ztmp)**2
         if (dd.gt.UPBOUND2) then
            fjs(i) = fjs(i)*UPBOUND2inv
            fjs(i-1) = fjs(i-1)*UPBOUND2inv
            iscale(i) = 1
         endif
 2200 continue
c
c ...  Step 3: go back up to the top and make sure that all
c              Bessel functions are scaled by the same factor
c              (i.e. the net total of times rescaling was invoked
c              on the way down in the previous loop).
c              At the same time, add scaling to fjs array.
c
      ncntr=0
      scalinv=done/scale
      sctot = 1.0d0
      do i=1,ntop
         sctot = sctot*scalinv
         if(iscale(i-1).eq.1) sctot=sctot*UPBOUND2inv
         fjs(i)=fjs(i)*sctot
      enddo
c
c ... Determine the normalization parameter:
c
c     From Abramowitz and Stegun (9.1.47) and (9.1.48), Euler's identity
c
        psi = 0d0
c
        if (dimag(z) .lt. 0) zmul = +ima
        if (dimag(z) .ge. 0) zmul = -ima
        zsn = zmul**(mod(ntop,4))
c
        zmulinv=1/zmul
        do i = ntop,1,-1
           psi = scale*psi+fjs(i)*zsn
           zsn = zsn*zmulinv
        enddo
        psi = 2*psi*scale+fjs(0)
c
        if (dimag(z) .lt. 0) zscale = cdexp(+ima*z) / psi
        if (dimag(z) .ge. 0) zscale = cdexp(-ima*z) / psi
c
c
c ... Scale the jfuns by zscale:
c
      ztmp=zscale
      do i=0,nterms
         fjs(i)=fjs(i)*ztmp
      enddo
c
c ... Finally, calculate the derivatives if desired:
c
      if (ifder.eq.1) then
         fjs(nterms+1)=fjs(nterms+1)*ztmp
c
         fjder(0)=-fjs(1)*scale
         do i=1,nterms
            dc1=0.5d0
            dc2=done-dc1
            dc1=dc1*scalinv
            dc2=dc2*scale
            fjder(i)=dc1*fjs(i-1)-dc2*fjs(i+1)
         enddo
      endif
      return
      end
c
