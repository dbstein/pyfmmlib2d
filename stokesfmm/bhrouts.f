c    This file contains basic subroutines for forming and
c    evaluating multipole expansions for Stokes flow
c    in 2D
c
c-----------------------------------------------------------------
c     Expansion formation subroutines
c
c     bh2dformp: creates multipole expansion due to collection of
c                 charge sources and dipoles for the biharmonic problem
c     bh2dformta: creates a local expansion due to a collection of
c                 charge sources and dipoles
c
c------------------------------------------------------------------     
c     Multipole and local translation operators
c
c     bh2dmpmp: Converts multipole expansion to a multipole expansion
c     bh2dmploc: Converts multipole expansion to local expansion
c     bh2dlocloc: converts local expansion to local expansion
c
c---------------------------------------------------------------------
c     Evaluation subroutines
c 
c     bh2dmpeval: Computes the complex velocity 
c                 due to a multipole expansion at a given target
c     bh2dmpevalall: Computes the complex velocity due to a multipole
c                    expansion at given target locations
c              
c     bh2dtaeval: Computes the complex velocity due to a 
c                 local expansion at a given target  
c     bh2dtaevalall: Computes the complex velocity due to a 
c                 local expansion at given target locations 
c     bh2ddireval: Direct calculation for a collection of charge
c                  sources and dipoles at a given target
c    
c*********************************************************************

c********************************************************************
c     EVALUATION SUBROUTINES
c********************************************************************

      subroutine bh2ddireval(sources,ns,ifc,charges,ifd,dippar1,
     1         dippar2,target,vel,ifgrada,grada,ifgradaa,gradaa)
c********************************************************************
c      This subroutine computes the complex velocity at the set
c      of nt targets at targets(2,nt) due to a collection of ns
c      charges and dipoles at sources(2,ns)
c      
c      In this subroutine the following rescaled versions of charges and
c      dipoles are used to compute the velocity field due to them
c
c      The complex velocity at zt = xt + i yt 
c      due to a charge a complex charge c1 at zs = xs+i ys is given by
c
c
c      vel = 2*c1 log|zt-zs| + c1_bar (zt-zs)/(zt_bar - zs_bar)
c            
c      The complex velocity due to a dipole with complex parameters
c      c2,c3 is given by
c
c
c      vel = c2/(zt-zs) - c2_bar (zt-zs)/(zt_bar-zs_bar)^2 + 
c             c3/(zt_bar - zs_bar)
c
c      The corresponding goursat functions (\phi \psi) for derivation 
c      of the complex velocity=\phi + z d/dz (\phi)_bar + \psi_bar
c
c      \phi(z) = c1 log (z-zs) + d1/(z-zs)
c      \psi(z) = d2_bar/(z-zs) + zs_bar d1/(z-zs)^2 + c1_bar log(z-zs)-
c               zs_bar c1 /(z-zs)
c
c      Analytic component of the gradient (grada)= d/dz (\phi(z))
c
c      Anti analytic component of the gradient (gradaa) = 
c      z (d^2/ dz^2 (\phi))_bar + (d/dz (\psi))_bar
c----------------------------------------------------------------------
c      INPUT parameters
c
c      sources(2,ns): location of the sources
c      ns           : number of sources
c      ifc          : flag for charges
c      charges      : charge strength
c      ifd          : flag for dipoles
c      dippar1      : dipole parameter 1 (corresponding to c2 in
c                     above expression)
c      dippar2      : dipole parameter 2 (corresponding to c3 in
c                     above expression)
c      target       : target location
c      ifgrada      : flag for computing analytic gradient
c      ifgradaa     : flag for computing anti analytic part of gradient
c--------------------------------------------------------------------
c      OUTPUT
c      vel          : Complex velocity at target
c      grada        : Complex analytic gradient 
c      gradaa       : Complex analytic antigradient
c
c--------------------------------------------------------------------

      implicit real *8 (a-h,o-z)
      integer ns,ifc,ifd,ifgrada,ifgradaa
      real *8 sources(2,ns), target(2)
      complex *16 vel,zs,zt,zdis,charges(*),dippar1(*),dippar2(*)
      complex *16 zdis1,zdis2
      complex *16 grada,gradaa,eye

      vel=dcmplx(0,0)
      grada=dcmplx(0,0)
      gradaa=dcmplx(0,0)

      eye = dcmplx(0,1.0d0)
      zt = dcmplx(target(1),target(2))
      do i=1,ns
         zs = dcmplx(sources(1,i),sources(2,i))
         zdis = zt-zs
         zdis1 = 1.0d0/zdis
         zdis2=zdis1**2
         if(ifc.eq.1) vel=vel+2*charges(i)*log(cdabs(zdis))+
     1                     dconjg(charges(i)*zdis1)*zdis
         if(ifd.eq.1) then
            vel=vel+dippar1(i)*zdis1 + dippar2(i)*dconjg(zdis1)
            vel=vel-dconjg(dippar1(i)*zdis2)*zdis
         endif
         if(ifgrada.eq.1) then
          if(ifc.eq.1) grada=grada+charges(i)*zdis1
          if(ifd.eq.1) grada=grada-dippar1(i)*(zdis2)
         endif
         if(ifgradaa.eq.1) then
          if(ifc.eq.1) then
            gradaa=gradaa+charges(i)*dconjg(zdis1)
            gradaa=gradaa-dconjg(charges(i)*zdis2)*zdis
          endif
          if(ifd.eq.1) then
            gradaa=gradaa-dippar2(i)*dconjg(zdis2)
            gradaa=gradaa+2*dconjg(dippar1(i)*zdis2*zdis1)*
     1                         zdis
           endif
         endif
      enddo
   
      return
      end

c*****************************************************************
      subroutine bh2dmpeval(rscale,center,mp1,mp2,mp3,mp4,mp5,nterms,
     1                     ztarg,vel,ifgrada,grada,ifgradaa,gradaa)
c********************************************************************
c     This subroutine evaluates the multipole expansion about
c     the "center" due to the multipole expansion
c     at the target z_targ
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k)/z^k + \sum mp2(k)/z_bar^k +
c            k                k
c      
c         z \sum mp3(k)/z_bar^k + Re(mp4(0) log(z) + \sum mp4(k)/z^k)
c              k                                       k
c
c          + i Re(mp5(0) log(z) + \sum mp5(k)/z^k)
c                                  k
c      z = (ztarg - center)/rscale
c
c      grada = gradient(analytic comoponent of vel)
c
c      gradaa = gradient(all but analytic component of vel)
c             = gradient(anti analytic + z times anti analytic 
c                         component of vel)
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation
c----------------------------------------------------------------------
c      INPUT parameters
c      rscale        : scaling parameter
c      center        : expansion center
c      mp1           : coeffs for multipole expansion - laurent
c      mp2           : coeffs for multipole expansion - antilaurent 
c      mp3           : coeffs for multipole expansion - z antilaurent
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c      nterms        : Number of terms in multipole expansion
c      ztarg         : target location
c      ifgrada       : flag controlling evaluation of analytic
c                      component of gradient
c                      ifgrada = 0, do not compute analytic gradient
c                      ifgrada = 1, compute analytic gradient 
c      ifgradaa      : flag contralling evaluaton of anti analytic 
c                      component of gradient
c                      ifgradaa = 0, do not compute anti analytic
c                                    gradient
c                      ifgradaa = 1, compute anti analytic gradient
c---------------------------------------------------------------------
c     OUTPUT
c     vel - Complex velocity
c     grada - gradient of analytic component (if requested)
c     gradaa - gradient of anti analytic component (if requested)
c*********************************************************************

      implicit real *8 (a-h,o-z)
      integer nterms,ifgrada,ifgradaa,nmax
      real *8 rscale,center(2),ztarg(2)
      complex *16 zc,zt,zdis,eye
      complex *16 mp1(0:nterms),mp2(0:nterms),mp3(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms),vel,ztemp,ztemp1
      complex *16 grada,gradaa,zdisinv,zpow(1:1000)


      nmax = nterms+3
      rinv=1/rscale
      zc = dcmplx(center(1),center(2))
      zt = dcmplx(ztarg(1),ztarg(2))
      zdis = zt - zc
      zdisinv = 1.0d0/zdis
      ztemp = rscale*zdisinv
      zpow(1)=ztemp
      do i=2,nmax
          zpow(i)=zpow(i-1)*ztemp
      enddo
      eye = dcmplx(0.0d0,1.0d0)
      vel = (mp4(0)+eye*mp5(0))*log(cdabs(zdis))
   
      do i=1,nterms
         vel = vel + mp1(i)*zpow(i) + mp2(i)*dconjg(zpow(i))
         vel = vel + mp3(i)*dconjg(zpow(i))*zdis
         vel = vel + dreal(mp4(i)*zpow(i))+eye*dreal(mp5(i)*zpow(i))
      enddo

      if(ifgrada.eq.1) then
           grada=0.5d0*(mp4(0)+eye*mp5(0))*zpow(1)
           do i=1,nterms
             grada=grada-mp1(i)*zpow(i+1)*i
             grada=grada-0.5*mp4(i)*i*zpow(i+1)
             grada=grada-eye*0.5*mp5(i)*i*zpow(i+1)
           enddo
           grada=grada*rinv
      endif
      
      if(ifgradaa.eq.1) then
          gradaa=0.5d0*(mp4(0)+eye*mp5(0))*dconjg(zpow(1))
          do i=1,nterms
             gradaa=gradaa-mp2(i)*dconjg(zpow(i+1))*i
             gradaa=gradaa-zdis*mp3(i)*dconjg(zpow(i+1))*i
             gradaa=gradaa-dconjg(0.5*mp4(i)*i*zpow(i+1))
             gradaa=gradaa+dconjg(eye*0.5*mp5(i)*i*zpow(i+1))
          enddo
          gradaa=gradaa*rinv
      endif
      
      return
      end
c********************************************************************
      subroutine bh2dmpevalall(rscale,center,mp1,mp2,mp3,mp4,mp5,
     1           nterms,ztarg,ntarg,vel,ifgrada,grada,ifgradaa,gradaa)
c********************************************************************
c     This subroutine evaluates the multipole expansion about
c     the "center" due to the multipole expansion
c     at targets ztarg(1:ntarg)
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k)/z^k + \sum mp2(k)/z_bar^k +
c            k                k
c      
c         z \sum mp3(k)/z_bar^k + Re(mp4(0) log(z) + \sum mp4(k)/z^k)
c              k                                       k
c
c          + i Re(mp5(0) log(z) + \sum mp5(k)/z^k)
c                                  k
c      z = (ztarg - center)/rscale
c
c      grada = gradient(analytic comoponent of vel)
c
c      gradaa = gradient(all but analytic component of vel)
c             = gradient(anti analytic + z times anti analytic 
c                         component of vel)
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation
c----------------------------------------------------------------------
c      INPUT parameters
c      rscale        : scaling parameter
c      center        : expansion center
c      mp1           : coeffs for multipole expansion - laurent
c      mp2           : coeffs for multipole expansion - antilaurent 
c      mp3           : coeffs for multipole expansion - z antilaurent
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c      nterms        : Number of terms in multipole expansion
c      ztarg         : target locations
c      ntarg         : number of targets
c      ifgrada       : flag controlling evaluation of analytic
c                      component of gradient
c                      ifgrada = 0, do not compute analytic gradient
c                      ifgrada = 1, compute analytic gradient 
c      ifgradaa      : flag contralling evaluaton of anti analytic 
c                      component of gradient
c                      ifgradaa = 0, do not compute anti analytic
c                                    gradient
c                      ifgradaa = 1, compute anti analytic gradient
c---------------------------------------------------------------------
c     OUTPUT
c     vel(1:ntarg) - Complex velocity
c     grada(2,1:ntarg) - gradient of analytic component (if requested)
c     gradaa(2,1:ntarg) - gradient of anti analytic component (if requested)
c*********************************************************************

      implicit real *8 (a-h,o-z)
      integer *4 ntarg,nterms,ifgrada,ifgradaa
      real *8 rscale,center(2),ztarg(2,ntarg)
      complex *16 vel(1),grada(1),gradaa(1),mp1(0:nterms),mp2(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms),gradatmp,gradaatmp,vtmp
      complex *16 mp3(0:nterms)
     
      do i=1,ntarg
         call bh2dmpeval(rscale,center,mp1,mp2,mp3,mp4,mp5,nterms,
     1    ztarg(1,i),vtmp,ifgrada,gradatmp,ifgradaa,gradaatmp)
        vel(i)=vel(i)+vtmp
        if(ifgrada.eq.1) then
           grada(i)=grada(i)+gradatmp
        endif        
        if(ifgradaa.eq.1) then
           gradaa(i)=gradaa(i)+gradaatmp
        endif        
      enddo

      return
      end
c********************************************************************
      subroutine bh2dtaeval(rscale,center,mp1,mp2,mp3,mp4,mp5,nterms,
     1                      ztarg,vel,ifgrada,grada,ifgradaa,gradaa)
c********************************************************************
c     This subroutine evaluates the multipole expansion about
c     the "center" due to the local expansion
c     at the target ztarg
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k) z^k + \sum mp2(k) z_bar^k +
c            k                k
c      
c         z \sum mp3(k) z_bar^k + Re(\sum mp4(k) z^k)
c              k                       k
c
c          + i Re( \sum mp5(k) z^k)
c                    k
c      z = (ztarg - center)/rscale
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation
c----------------------------------------------------------------------
c      INPUT parameters
c      rscale        : scaling parameter
c      center        : expansion center
c      mp1           : coeffs for multipole expansion - taylor
c      mp2           : coeffs for multipole expansion - antitaylor 
c      mp3           : coeffs for multipole expansion - z antitaylor
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c      nterms        : Number of terms in multipole expansion
c      ztarg         : target location
c      ifgrada       : flag for computing the analytic gradient 
c                       (d/dz (phi))
c      ifgradaa      : flag for computing the anti analytic gradient
c                      (d/dz_bar (z (d/dz phi)_bar+\psi_bar)
c---------------------------------------------------------------------
c     OUTPUT
c     vel - Complex velocity
c*********************************************************************

      implicit real *8 (a-h,o-z)
      integer nterms,ifgrada,ifgradaa
      real *8 rscale,center(2),ztarg(2)
      complex *16 zc,zt,zdis,eye
      complex *16 mp1(0:nterms),mp2(0:nterms),mp3(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms),vel,ztemp,ztemp1
      complex *16 grada,gradaa,zpow(0:1000)

      rinv=1/rscale
      zt = dcmplx(ztarg(1),ztarg(2))
      zc = dcmplx(center(1),center(2))
      zdis = zt - zc
      ztemp1 = zdis*rinv
      zpow(0)=1.0d0
      do i=1,nterms
         zpow(i)=zpow(i-1)*ztemp1
      enddo
      eye = dcmplx(0.0d0,1.0d0)
      
      vel = dcmplx(0.0d0,0.0d0)
      grada=dcmplx(0.0d0,0.0d0)
      gradaa=dcmplx(0.0d0,0.0d0)
      do i=0,nterms
         vel = vel + mp1(i)*zpow(i) + mp2(i)*dconjg(zpow(i))
         vel = vel + mp3(i)*dconjg(zpow(i))*zdis
         vel = vel + dreal(mp4(i)*zpow(i))+eye*dreal(mp5(i)*zpow(i))
      enddo
      if(ifgrada.eq.1) then
            do i=1,nterms
               grada=grada+mp1(i)*zpow(i-1)*i
               grada=grada+0.5*mp4(i)*i*zpow(i-1)
               grada=grada+eye*0.5*mp5(i)*i*zpow(i-1)
            enddo
            grada=grada*rinv
      endif

      if(ifgradaa.eq.1) then
            do i=1,nterms
             gradaa=gradaa+mp2(i)*dconjg(zpow(i-1))*i
             gradaa=gradaa+mp3(i)*dconjg(zpow(i-1))*i*zdis
             gradaa=gradaa+dconjg(0.5*mp4(i)*i*zpow(i-1))
             gradaa=gradaa-dconjg(eye*0.5*mp5(i)*i*zpow(i-1))
            enddo
            gradaa=gradaa*rinv
      endif
   
      return
      end
c********************************************************************
      subroutine bh2dtaevalall(rscale,center,mp1,mp2,mp3,mp4,mp5,
     1         nterms,ztarg,ntarg,vel,ifgrada,grada,ifgradaa,gradaa)
c********************************************************************
c     This subroutine evaluates the multipole expansion about
c     the "center" due to the local expansion
c     at the target locations ztarg(1:ntarg)
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k) z^k + \sum mp2(k) z_bar^k +
c            k                k
c      
c         z \sum mp3(k) z_bar^k + Re(\sum mp4(k) z^k)
c              k                       k
c
c          + i Re( \sum mp5(k) z^k)
c                    k
c      z = (ztarg - center)/rscale
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation
c----------------------------------------------------------------------
c      INPUT parameters
c      rscale        : scaling parameter
c      center        : expansion center
c      mp1           : coeffs for multipole expansion - taylor
c      mp2           : coeffs for multipole expansion - antitaylor 
c      mp3           : coeffs for multipole expansion - z antitaylor
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c      nterms        : Number of terms in multipole expansion
c      ztarg         : target locations
c      ntarg         : number of target locations
c      ifgrada       : flag for computing the analytic gradient 
c                       (d/dz (phi))
c      ifgradaa      : flag for computing the anti analytic gradient
c                      (d/dz_bar (z (d/dz phi)_bar+\psi_bar)
c---------------------------------------------------------------------
c     OUTPUT
c     vel(1:ntarg) - Complex velocity
c     grada(2,1:ntarg) - Analytic gradient
c     gradaa(2,1:ntarg) - Antianalytic gradient
c*********************************************************************

      implicit real *8 (a-h,o-z)
      integer *4 ntarg,nterms,ifgrada,ifgradaa
      real *8 rscale,center(2),ztarg(2,ntarg)
      complex *16 vel(1),grada(1),gradaa(1),mp1(0:nterms),mp2(0:nterms)
      complex *16 mp3(0:nterms),mp4(0:nterms),mp5(0:nterms),gradatmp
      complex *16 gradaatmp,vtmp
     
      do i=1,ntarg
         call bh2dtaeval(rscale,center,mp1,mp2,mp3,mp4,mp5,nterms,
     1    ztarg(1,i),vtmp,ifgrada,gradatmp,ifgradaa,gradaatmp)

        vel(i)=vel(i)+vtmp
        if(ifgrada.eq.1) then
           grada(i)=grada(i)+gradatmp
        endif        
        if(ifgradaa.eq.1) then
           gradaa(i)=gradaa(i)+gradaatmp
        endif
      enddo

      return
      end

c*******************************************************************
c     EXPANSION FORMATION
c*******************************************************************
      subroutine bh2dformmp(ier,rscale,sources,ifc,c1,ifd,d1,d2,
     1           ns,center,nterms,mp1,mp2,mp3,mp4,mp5)
c*****************************************************************
c     This subroutine computes the multipole expansion about
c     the "center" due to ns sources located at sources(2,*)
c
c      The complex velocity at zt = xt + i yt 
c      due to a charge a complex charge c1 at zs = xs+i ys is given by
c
c
c      vel = 2*c1 log|zt-zs| + c1_bar (zt-zs)/(zt_bar - zs_bar)
c            
c      The complex velocity due to a dipole with complex parameters
c      c2,c3 is given by
c
c
c      vel = c2/(zt-zs) - c2_bar (zt-zs)/(zt_bar-zs_bar)^2 + 
c             c3/(zt_bar - zs_bar)
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k)/z^k + \sum mp2(k)/z_bar^k +
c            k                k
c      
c         z \sum mp3(k)/z_bar^k + Re(mp4(0) log(z) + \sum mp4(k)/z^k)
c              k                                       k
c
c          + i Re(mp5(0) log(z) + \sum mp5(k)/z^k)
c                                  k
c------------------------------------------------------------------
c     INPUT parameters
c
c      rscale        : scaling parameter
c      sources(2,ns) : coordinates of sources
c      ifc          : flag for charges
c      c1           : charge strength
c      ifd          : flag for dipoles
c      d1           : dipole parameter 1 (corresponding to c2)
c      d2           : dipole parameter 2 (corresponding to c3)
c      ns           : number of sources
c      center(2)    : expansion center
c      nterms       : Order of multipole expansion
c------------------------------------------------------------------
c      OUTPUT
c
c      ier           : error return code
c                      ier=0 returned successfully
c      mp1           : coeffs for multipole expansion - laurent
c      mp2           : coeffs for multipole expansion - antilaurent 
c      mp3           : coeffs for multipole expansion - z antilaurent
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c-----------------------------------------------------------------

      implicit real *8 (a-h,o-z)
      integer ns,nterms,ifc,ifd,ier
      complex *16 mp1(0:nterms),mp2(0:nterms),mp3(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms)
      complex *16 c1(*),d1(*),d2(*)
      real *8 sources(2,ns),center(2),rscale
      complex *16 mptemp1(0:nterms),mptemp2(0:nterms)
      complex *16 zc,z1,z2,zs,zdis,ztemp,zdisc,ztempc
      complex *16 ztempc2
      complex *16 zt1,zt2,zt3

      ier = 0
      zc = dcmplx(center(1),center(2))
      rinv=1.0d0/rscale
      do i=0,nterms
         mp1(i)=0.0d0
         mp2(i)=0.0d0
         mp3(i)=0.0d0
         mp4(i)=0.0d0
         mp5(i)=0.0d0
      enddo
    
      do i=1,ns
         zs=dcmplx(sources(1,i),sources(2,i))
         zdis = (zs-zc)/rscale
         ztemp= zs-zc
         zdisc=dconjg(zdis)
         ztempc=1.0d0/dconjg(ztemp)
         ztempc2=ztempc*ztempc
         if(ifc.eq.1) then
            mp4(0)=mp4(0)+dreal(2*c1(i))
            mp5(0)=mp5(0)+dimag(2*c1(i))
         endif
         do j=1,nterms
            if(ifc.eq.1) then
               zt1 = dconjg(c1(i))*ztempc
c       Expansion corresponding to 2 c1 log |z-zs|
              mp4(j)=mp4(j)-dreal(2*c1(i))*zdis/j
              mp5(j)=mp5(j)-dimag(2*c1(i))*zdis/j
c       Expansion corresponding to c1_bar z-zs/(z_bar - zs_bar)
              mp2(j)=mp2(j)-zt1*zdisc*ztemp
              mp3(j)=mp3(j)+zt1*zdisc
            endif

            if(ifd.eq.1) then
               zt2 = d2(i)*ztempc
               zt3 = dconjg(d1(i))*ztempc2
c        Expansion corresponding to d1/z-zs
               mp1(j)=mp1(j)+d1(i)*zdis/ztemp
c        Expansion corresponding to d2/(z_bar - zs_bar)
               mp2(j)=mp2(j)+zt2*zdisc
c        Expansion corresponding to -d1_bar(z-zs)/(z_bar - zs_bar)^2
               if(j.gt.1) then
                 mp2(j)=mp2(j)+zt3*(j-1)*zdisc*ztemp
                 mp3(j)=mp3(j)-zt3*(j-1)*zdisc
               endif
            endif 
            zdis=zdis*ztemp*rinv
            zdisc=zdisc/ztempc*rinv
         enddo
      enddo

      return
      end

c********************************************************************
      subroutine bh2dformta(ier,rscale,sources,ifc,c1,ifd,d1,d2,
     1       ns,center,nterms,mp1,mp2,mp3,mp4,mp5)
c*****************************************************************
c     This subroutine computes the local expansion about
c     the "center" due to ns sources located at sources(2,*)
c
c      The complex velocity at zt = xt + i yt 
c      due to a charge a complex charge c1 at zs = xs+i ys is given by
c
c
c      vel = 2*c1 log|zt-zs| + c1_bar (zt-zs)/(zt_bar - zs_bar)
c            
c      The complex velocity due to a dipole with complex parameters
c      c2,c3 is given by
c
c
c      vel = c2/(zt-zs) - c2_bar (zt-zs)/(zt_bar-zs_bar)^2 + 
c             c3/(zt_bar - zs_bar)
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k) z^k + \sum mp2(k) z_bar^k +
c            k                k
c      
c         z \sum mp3(k) z_bar^k + Re(mp4(0) log(z) + \sum mp4(k) z^k)
c              k                                       k
c
c          + i Re(mp5(0) log(z) + \sum mp5(k) z^k)
c                                  k
c------------------------------------------------------------------
c     INPUT parameters
c
c      rscale        : scaling parameter
c      sources(2,ns) : coordinates of sources
c      ifc          : flag for charges
c      c1           : charge strength
c      ifd          : flag for dipoles
c      d1           : dipole parameter 1 (corresponding to c2)
c      d2           : dipole parameter 2 (corresponding to c3)
c      ns            : number of sources
c      center(2)     : expansion center
c      nterms        : Order of multipole expansion
c------------------------------------------------------------------
c      OUTPUT
c
c      ier           : error return code
c                      ier=0 returned successfully
c      mp1           : coeffs for multipole expansion - laurent
c      mp2           : coeffs for multipole expansion - antilaurent 
c      mp3           : coeffs for multipole expansion - z antilaurent
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c-----------------------------------------------------------------

      implicit real *8 (a-h,o-z)
      integer ns,nterms,ifc,ifd,ier
      complex *16 mp1(0:nterms),mp2(0:nterms),mp3(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms)
      complex *16 c1(*),d1(*),d2(*)
      real *8 sources(2,ns),center(2),rscale
      complex *16 mptemp1(0:nterms),mptemp2(0:nterms)
      complex *16 zc,z1,z2,zs,zdis,ztemp,zdisc,ztempc

      ier = 0
      zc = dcmplx(center(1),center(2))
      do i=0,nterms
         mp1(i)=0.0d0
         mp2(i)=0.0d0
         mp3(i)=0.0d0
         mp4(i)=0.0d0
         mp5(i)=0.0d0
      enddo

      do i=1,ns
         zs=dcmplx(sources(1,i),sources(2,i))
         zdis = 1.0d0
         ztemp= 1.0d0/(zs-zc)
         zdisc=dconjg(zdis)
         ztempc=dconjg(ztemp)
         do j=0,nterms
            if(ifc.eq.1) then
              if(j.eq.0) then
                 mp4(j)=mp4(j)+dreal(2*c1(i))*log(cdabs(1.0d0/ztemp))
                 mp5(j)=mp5(j)+dimag(2*c1(i))*log(cdabs(1.0d0/ztemp))
              else
                mp4(j)=mp4(j)-dreal(2*c1(i))*zdis/j
                mp5(j)=mp5(j)-dimag(2*c1(i))*zdis/j
              endif
              mp2(j)=mp2(j)+dconjg(c1(i))*zdisc*ztempc/ztemp
              mp3(j)=mp3(j)-dconjg(c1(i))*zdisc*ztempc
            endif
 
            if(ifd.eq.1) then
               mp1(j)=mp1(j)-d1(i)*zdis*ztemp
               mp2(j)=mp2(j)-d2(i)*zdisc*ztempc
               mp2(j)=mp2(j)+dconjg(d1(i))*(j+1)*zdisc*
     1                       ztempc*ztempc/ztemp
               mp3(j)=mp3(j)-dconjg(d1(i))*(j+1)*zdisc*ztempc*ztempc
            
            endif

            zdis=zdis*ztemp*rscale
            zdisc=zdisc*ztempc*rscale
         enddo
      enddo

      return
      end

c********************************************************************
      subroutine bh2dformta_add(ier,rscale,sources,ifc,c1,ifd,d1,d2,
     1       ns,center,nterms,mp1,mp2,mp3,mp4,mp5)
c*****************************************************************
c     This subroutine computes the local expansion about
c     the "center" due to ns sources located at sources(2,*)
c
c      The complex velocity at zt = xt + i yt 
c      due to a charge a complex charge c1 at zs = xs+i ys is given by
c
c
c      vel = 2*c1 log|zt-zs| + c1_bar (zt-zs)/(zt_bar - zs_bar)
c            
c      The complex velocity due to a dipole with complex parameters
c      c2,c3 is given by
c
c
c      vel = c2/(zt-zs) - c2_bar (zt-zs)/(zt_bar-zs_bar)^2 + 
c             c3/(zt_bar - zs_bar)
c
c     The multipole expansion has 5 set of coefficients
c     vel = \sum mp1(k) z^k + \sum mp2(k) z_bar^k +
c            k                k
c      
c         z \sum mp3(k) z_bar^k + Re(mp4(0) log(z) + \sum mp4(k) z^k)
c              k                                       k
c
c          + i Re(mp5(0) log(z) + \sum mp5(k) z^k)
c                                  k
c------------------------------------------------------------------
c     INPUT parameters
c
c      rscale        : scaling parameter
c      sources(2,ns) : coordinates of sources
c      ifc          : flag for charges
c      c1           : charge strength
c      ifd          : flag for dipoles
c      d1           : dipole parameter 1 (corresponding to c2)
c      d2           : dipole parameter 2 (corresponding to c3)
c      ns            : number of sources
c      center(2)     : expansion center
c      nterms        : Order of multipole expansion
c------------------------------------------------------------------
c      OUTPUT
c
c      ier           : error return code
c                      ier=0 returned successfully
c      mp1           : coeffs for multipole expansion - laurent
c      mp2           : coeffs for multipole expansion - antilaurent 
c      mp3           : coeffs for multipole expansion - z antilaurent
c      mp4           : coeffs for multipole expansion - logsource1
c      mp5           : coeffs for multipole expansion - logsource2
c-----------------------------------------------------------------

      implicit real *8 (a-h,o-z)
      integer ns,nterms,ifc,ifd,ier
      complex *16 mp1(0:nterms),mp2(0:nterms),mp3(0:nterms)
      complex *16 mp4(0:nterms),mp5(0:nterms)
      complex *16 c1(ns),d1(ns),d2(ns)
      real *8 sources(2,ns),center(2),rscale
      complex *16 mptemp1(0:nterms),mptemp2(0:nterms)
      complex *16 zc,z1,z2,zs,zdis,ztemp,zdisc,ztempc

      ier = 0
      zc = dcmplx(center(1),center(2))

      do i=1,ns
         zs=dcmplx(sources(1,i),sources(2,i))
         zdis = 1.0d0
         ztemp= 1.0d0/(zs-zc)
         zdisc=dconjg(zdis)
         ztempc=dconjg(ztemp)
         do j=0,nterms
            if(ifc.eq.1) then
              if(j.eq.0) then
                 mp4(j)=mp4(j)+dreal(2*c1(i))*log(cdabs(1.0d0/ztemp))
                 mp5(j)=mp5(j)+dimag(2*c1(i))*log(cdabs(1.0d0/ztemp))
              else
                mp4(j)=mp4(j)-dreal(2*c1(i))*zdis/j
                mp5(j)=mp5(j)-dimag(2*c1(i))*zdis/j
              endif
              mp2(j)=mp2(j)+dconjg(c1(i))*zdisc*ztempc/ztemp
              mp3(j)=mp3(j)-dconjg(c1(i))*zdisc*ztempc
            endif
 
            if(ifd.eq.1) then
               mp1(j)=mp1(j)-d1(i)*zdis*ztemp
               mp2(j)=mp2(j)-d2(i)*zdisc*ztempc
               mp2(j)=mp2(j)+dconjg(d1(i))*(j+1)*zdisc*
     1                       ztempc*ztempc/ztemp
               mp3(j)=mp3(j)-dconjg(d1(i))*(j+1)*zdisc*ztempc*ztempc
            
            endif

            zdis=zdis*ztemp*rscale
            zdisc=zdisc*ztempc*rscale
         enddo
      enddo

      return
      end

c********************************************************************
c     TRANSLATION OPERATORS
c*******************************************************************

      subroutine bh2dlocloc(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2)
c******************************************************************
c     Converts local expansion to local expansion
c     Given original expansions
c
c     Given local expansion in the following form
c     vel = \sum h1(k) z1^k + \sum h2(k) z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter for original expansion 
c     c1      : center of original local expansion
c     h1      : coefficients of original taylor expansion
c     h2      : coefficients of original antitaylor expansion
c     h3      : coefficients of original z1 (antitaylor expansion)
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c
c     nterms1 : number of terms in original expansion
c     rscale2 : scaling parameter for shifted expansion
c     c2      : center of new local expansion
c     nterms2 : number of terms of the shifted expansion
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted taylor expansion
c     j2      : coefficients of shifted antitaylor expansion
c     j3      : coefficients of shifted z1 (antitaylor expansion)
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zpow1(0:1000),zpow2(0:1000),zdis,zdisinv
      real *8 carray(0:1000,0:1000),rscale1,rscale2,c1(2),c2(2)

c     Constructing the binomial coefficients for the transform
      do i=0,nterms1
          carray(i,0)=1.0d0
      enddo
      
      do i=1,nterms1
          carray(i,i)=1.0d0
          do j=i+1,nterms1
             carray(j,i)=carray(j-1,i)+carray(j-1,i-1)
          enddo
      enddo
     
      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = zc2-zc1
      zdisinv = 1.0d0/zdis
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      rinv1 = 1.0d0/rscale1
      do i=1,nterms1
           zpow1(i)=zpow1(i-1)*zdis*rinv1
           zpow2(i)=zpow2(i-1)*rscale2*zdisinv
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

      do i=0,max(nterms1,nterms2)
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)+h3(i)*zdis)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
      enddo

      do i=0,nterms2
         do j=i,nterms1
             j1(i)=j1(i)+h1a(j)*carray(j,i)
             j2(i)=j2(i)+h2a(j)*carray(j,i)
             j3(i)=j3(i)+h3a(j)*carray(j,i)
             j4(i)=j4(i)+h4a(j)*carray(j,i)
             j5(i)=j5(i)+h5a(j)*carray(j,i)
        enddo
             j1(i)=j1(i)*zpow2(i)
             j2(i)=j2(i)*dconjg(zpow2(i))
             j3(i)=j3(i)*dconjg(zpow2(i))
             j4(i)=j4(i)*zpow2(i)
             j5(i)=j5(i)*zpow2(i)
      enddo

      return
      end
c******************************************************************
      subroutine bh2dmpmp(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2)
c******************************************************************
c     Converts multipole expansion to multipole expansion
c     Given original expansions
c
c     Given multipole expansion in the following form
c     vel = \sum h1(k) /z1^k + \sum h2(k) /z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) /z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) /z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) /z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation
c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter of original multipole expansion
c     c1      : center of original mutlipole expansion
c     h1      : coefficients of original laurent expansion
c     h2      : coefficients of original antilaurent expansion
c     h3      : coefficients of original z1 *antilaurent expansion
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms in original expansion
c     rscale2 : scaling parameter of shifted multipole expansion
c     c2      : center of new multipole expansion
c     nterms2 : number of terms of the shifted expansion
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted laurent expansion
c     j2      : coefficients of shifted antilaurent expansion
c     j3      : coefficients of shifted z1 *antilaurent expansion
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zpow1(0:1000),zdis,zdisinv
      complex *16 zpow2(0:1000)
      real *8 carray(0:1000,0:1000)
      real *8 rscale1,rscale2,c1(2),c2(2),rinv2

c     Constructing the binomial coefficients for the transform
      nmax=max(nterms1,nterms2)
      do i=0,nmax
          carray(i,0)=1.0d0
      enddo
      
      do i=1,nmax
          carray(i,i)=1.0d0
          do j=i+1,nmax
             carray(j,i)=carray(j-1,i)+carray(j-1,i-1)
          enddo
      enddo

      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = zc1-zc2
      zdisinv = 1.0d0/zdis
      rinv2 = 1.0d0/rscale2
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      do i=1,nmax
           zpow1(i)=zpow1(i-1)*zdisinv*rscale1
           zpow2(i)=zpow2(i-1)*zdis*rinv2
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

c     Handling the log term in the expansion
      j4(0)=h4(0)
      j5(0)=h5(0)
      do i=1,nterms2
          j4(i)=j4(i)-h4(0)/i
          j5(i)=j5(i)-h5(0)/i
      enddo

      do i=1,min(nterms2,nterms1)
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)-h3(i)*zdis)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
      enddo

      do i=1,nterms2
         do j=1,min(i,nterms1)
             j1(i)=j1(i)+h1a(j)*carray(i-1,j-1)
             j2(i)=j2(i)+h2a(j)*carray(i-1,j-1)
             j3(i)=j3(i)+h3a(j)*carray(i-1,j-1)
             j4(i)=j4(i)+h4a(j)*carray(i-1,j-1)
             j5(i)=j5(i)+h5a(j)*carray(i-1,j-1)
        enddo
        j1(i)=j1(i)*zpow2(i)
        j2(i)=j2(i)*dconjg(zpow2(i))
        j3(i)=j3(i)*dconjg(zpow2(i))
        j4(i)=j4(i)*zpow2(i)
        j5(i)=j5(i)*zpow2(i)
      enddo

      return
      end
c*******************************************************************
      subroutine bh2dmploc(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2)
c******************************************************************
c     Converts multipole expansion to local expansion
c     Given original expansions
c
c     Given multipole expansion in the following form
c     vel = \sum h1(k) /z1^k + \sum h2(k) /z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) /z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) /z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) /z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter of original multipole expansion 
c     c1      : center of original mutlipole expansion
c     h1      : coefficients of original laurent expansion
c     h2      : coefficients of original antilaurent expansion
c     h3      : coefficients of original z1 *antilaurent expansion
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms of original expansion
c     rscale2 : scaling parameter of shifted local expansion
c     c2      : center of new local expansion
c     nterms2 : number of terms of the shifted expansion
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted taylor expansion
c     j2      : coefficients of shifted antaylor expansion
c     j3      : coefficients of shifted z1 *antitaylor expansion
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zdis,zdis1
      complex *16 zpow1(0:1000)
      complex *16 zzz,zzz1
      complex *16 zpow2(0:1000)
      real *8 carray(0:1000,0:1000)
      real *8 rscale1,rscale2,c1(2),c2(2)

c     Constructing the binomial coefficients for the transform
      nmax=nterms1+nterms2
      do i=0,nmax
          carray(i,0)=1.0d0
      enddo
      
      do i=1,nmax
          carray(i,i)=1.0d0
          do j=i+1,nmax
             carray(j,i)=carray(j-1,i)+carray(j-1,i-1)
          enddo
      enddo
      
      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = 1/(zc1-zc2)
      zdis1 = (zc1-zc2)
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      do i=1,nmax
           zpow1(i)=-zpow1(i-1)*zdis*rscale1
           zpow2(i)=zpow2(i-1)*zdis*rscale2
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

c     Handling the log term in the expansion
      j4(0)=h4(0)*log(cdabs(zdis1))
      j5(0)=h5(0)*log(cdabs(zdis1))
      do i=1,nterms2
          j4(i)=j4(i)-h4(0)/i
          j5(i)=j5(i)-h5(0)/i
      enddo
c
      do i=1,nterms1
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)-h3(i)*zdis1)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
ccc          h2a(i)=h2a(i)-h3a(i)*zdis1
      enddo
      do i=0,nterms2
         do j=1,nterms1
             j1(i)=j1(i)+h1a(j)*carray(i+j-1,i)
             j2(i)=j2(i)+h2a(j)*carray(i+j-1,i)
             j3(i)=j3(i)+h3a(j)*carray(i+j-1,i)
             j4(i)=j4(i)+h4a(j)*carray(i+j-1,i)
             j5(i)=j5(i)+h5a(j)*carray(i+j-1,i)
        enddo
        j1(i)=j1(i)*zpow2(i)
        j2(i)=j2(i)*dconjg(zpow2(i))
        j3(i)=j3(i)*dconjg(zpow2(i))
        j4(i)=j4(i)*zpow2(i)
        j5(i)=j5(i)*zpow2(i)
      enddo
      
      return
      end
c*******************************************************************
      subroutine bh2dmploc_add(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2)
c******************************************************************
c     Converts multipole expansion to local expansion
c     Given original expansions
c
c     Given multipole expansion in the following form
c     vel = \sum h1(k) /z1^k + \sum h2(k) /z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) /z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) /z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) /z1^k)
c     z1 = z-c1
c

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter of original multipole expansion 
c     c1      : center of original mutlipole expansion
c     h1      : coefficients of original laurent expansion
c     h2      : coefficients of original antilaurent expansion
c     h3      : coefficients of original z1 *antilaurent expansion
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms of original expansion
c     rscale2 : scaling parameter of shifted local expansion
c     c2      : center of new local expansion
c     nterms2 : number of terms of the shifted expansion
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted taylor expansion
c     j2      : coefficients of shifted antaylor expansion
c     j3      : coefficients of shifted z1 *antitaylor expansion
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zpow1(0:nterms1+nterms2),zdis
      complex *16 zpow2(0:nterms1+nterms2)
      real *8 carray(0:nterms1+nterms2,0:nterms1+nterms2)
      real *8 rscale1,rscale2,c1(2),c2(2)

c     Constructing the binomial coefficients for the transform
      nmax=nterms1+nterms2
      do i=0,nmax
          carray(i,0)=1.0d0
      enddo
      
      do i=1,nmax
          carray(i,i)=1.0d0
          do j=i+1,nmax
             carray(j,i)=carray(j-1,i)+carray(j-1,i-1)
          enddo
      enddo
      
      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = 1/(zc1-zc2)
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      do i=1,nmax
           zpow1(i)=zpow1(i-1)*zdis*rscale1
           zpow2(i)=zpow2(i-1)*zdis*rscale2
      enddo
      
c     Handling the log term in the expansion
      j4(0)=h4(0)*log(cdabs(1.0d0/zdis))
      j5(0)=h5(0)*log(cdabs(1.0d0/zdis))
      do i=1,nterms2
          j4(i)=j4(i)-h4(0)*zpow2(i)/i
          j5(i)=j5(i)-h5(0)*zpow2(i)/i
      enddo
      do i=0,nterms2
         s=-1.0d0
         do j=1,nterms1
             j1(i)=j1(i)+h1(j)*carray(i+j-1,i)*zpow2(i)*zpow1(j)*s
             j2(i)=j2(i)+h2(j)*carray(i+j-1,i)*dconjg(zpow2(i)*zpow1(j))
     1                         *s
             j2(i)=j2(i)-h3(j)*carray(i+j-1,i)*dconjg(zpow2(i)*zpow1(j))
     1                       /zdis*s
             j3(i)=j3(i)+h3(j)*carray(i+j-1,i)*dconjg(zpow2(i)*zpow1(j))
     1                          *s
             j4(i)=j4(i)+h4(j)*carray(i+j-1,i)*zpow2(i)*zpow1(j)*s
             j5(i)=j5(i)+h5(j)*carray(i+j-1,i)*zpow2(i)*zpow1(j)*s
             s=-s
        enddo
        
      enddo

      return
      end
c*******************************************************************

      subroutine bh2dlocloc_carray(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2,carray,ldc)
c******************************************************************
c     Converts local expansion to local expansion
c     Given original expansions and precomputed binomial coefficients
c
c     Given local expansion in the following form
c     vel = \sum h1(k) z1^k + \sum h2(k) z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter for original expansion 
c     c1      : center of original local expansion
c     h1      : coefficients of original taylor expansion
c     h2      : coefficients of original antitaylor expansion
c     h3      : coefficients of original z1 (antitaylor expansion)
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms in original expansion
c     rscale2 : scaling parameter for shifted expansion
c     c2      : center of new local expansion
c     nterms2 : number of terms of the shifted expansion
c     carray  : Precomputed binomial coefficients
c     ldc     : size of carray
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted taylor expansion
c     j2      : coefficients of shifted antitaylor expansion
c     j3      : coefficients of shifted z1 (antitaylor expansion)
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zpow1(0:1000),zpow2(0:1000),zdis,zdisinv
      real *8 carray(0:ldc,0:ldc),rscale1,rscale2,c1(2),c2(2)

      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = zc2-zc1
      zdisinv = 1.0d0/zdis
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      rinv1 = 1.0d0/rscale1
      do i=1,nterms1
           zpow1(i)=zpow1(i-1)*zdis*rinv1
           zpow2(i)=zpow2(i-1)*rscale2*zdisinv
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

      do i=0,max(nterms1,nterms2)
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)+h3(i)*zdis)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
      enddo

      do i=0,nterms2
         do j=i,nterms1
             j1(i)=j1(i)+h1a(j)*carray(j,i)
             j2(i)=j2(i)+h2a(j)*carray(j,i)
             j3(i)=j3(i)+h3a(j)*carray(j,i)
             j4(i)=j4(i)+h4a(j)*carray(j,i)
             j5(i)=j5(i)+h5a(j)*carray(j,i)
        enddo
             j1(i)=j1(i)*zpow2(i)
             j2(i)=j2(i)*dconjg(zpow2(i))
             j3(i)=j3(i)*dconjg(zpow2(i))
             j4(i)=j4(i)*zpow2(i)
             j5(i)=j5(i)*zpow2(i)
      enddo

      return
      end
c******************************************************************
      subroutine bh2dmpmp_carray(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2,carray,ldc)
c******************************************************************
c     Converts multipole expansion to multipole expansion
c     Given original expansions and precomputed binomial coefficients
c
c     Given multipole expansion in the following form
c     vel = \sum h1(k) /z1^k + \sum h2(k) /z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) /z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) /z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) /z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter of original multipole expansion
c     c1      : center of original mutlipole expansion
c     h1      : coefficients of original laurent expansion
c     h2      : coefficients of original antilaurent expansion
c     h3      : coefficients of original z1 *antilaurent expansion
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms in original expansion
c     rscale2 : scaling parameter of shifted multipole expansion
c     c2      : center of new multipole expansion
c     nterms2 : number of terms of the shifted expansion
c     carray  : precomputed binomial coefficieints array
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted laurent expansion
c     j2      : coefficients of shifted antilaurent expansion
c     j3      : coefficients of shifted z1 *antilaurent expansion
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zpow1(0:1000),zdis,zdisinv
      complex *16 zpow2(0:1000)
      real *8 carray(0:ldc,0:ldc)
      real *8 rscale1,rscale2,c1(2),c2(2),rinv2

      nmax=max(nterms1,nterms2)
      
      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = zc1-zc2
      zdisinv = 1.0d0/zdis
      rinv2 = 1.0d0/rscale2
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      do i=1,nmax
           zpow1(i)=zpow1(i-1)*zdisinv*rscale1
           zpow2(i)=zpow2(i-1)*zdis*rinv2
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

c     Handling the log term in the expansion
      j4(0)=h4(0)
      j5(0)=h5(0)
      do i=1,nterms2
          j4(i)=j4(i)-h4(0)/i
          j5(i)=j5(i)-h5(0)/i
      enddo

      do i=1,min(nterms2,nterms1)
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)-h3(i)*zdis)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
      enddo

      do i=1,nterms2
         do j=1,min(i,nterms1)
             j1(i)=j1(i)+h1a(j)*carray(i-1,j-1)
             j2(i)=j2(i)+h2a(j)*carray(i-1,j-1)
             j3(i)=j3(i)+h3a(j)*carray(i-1,j-1)
             j4(i)=j4(i)+h4a(j)*carray(i-1,j-1)
             j5(i)=j5(i)+h5a(j)*carray(i-1,j-1)
        enddo
        j1(i)=j1(i)*zpow2(i)
        j2(i)=j2(i)*dconjg(zpow2(i))
        j3(i)=j3(i)*dconjg(zpow2(i))
        j4(i)=j4(i)*zpow2(i)
        j5(i)=j5(i)*zpow2(i)
      enddo

      return
      end
c*******************************************************************
      subroutine bh2dmploc_carray(rscale1,c1,h1,h2,h3,h4,h5,nterms1,
     1       rscale2,c2,j1,j2,j3,j4,j5,nterms2,carray,ldc)
c******************************************************************
c     Converts multipole expansion to local expansion
c     Given original expansions and precomputed binomial coeffients
c
c     Given multipole expansion in the following form
c     vel = \sum h1(k) /z1^k + \sum h2(k) /z1_bar^k +
c            k                k
c      
c     z1 \sum h3(k) /z1_bar^k + Re(h4(0) log(z1) + \sum h4(k) /z1^k)
c              k                                       k
c
c          + i Re(h5(0) log(z1) + \sum h5(k) /z1^k)
c     z1 = z-c1
c
c     NOTE: The subroutine will work if nterms<1000, as 1000, is
c     a hardwired number for computing zpow in the computation

c-----------------------------------------------------------------
c     INPUT
c     rscale1 : scaling parameter of original multipole expansion 
c     c1      : center of original mutlipole expansion
c     h1      : coefficients of original laurent expansion
c     h2      : coefficients of original antilaurent expansion
c     h3      : coefficients of original z1 *antilaurent expansion
c     h4      : coefficients of original real part logsource1
c     h5      : coefficients of original imag part logsource2
c     nterms1 : number of terms of original expansion
c     rscale2 : scaling parameter of shifted local expansion
c     c2      : center of new local expansion
c     nterms2 : number of terms of the shifted expansion
c-----------------------------------------------------------------
c     OUTPUT
c     j1      : coefficients of shifted taylor expansion
c     j2      : coefficients of shifted antaylor expansion
c     j3      : coefficients of shifted z1 *antitaylor expansion
c     j4      : coefficients of shifted real part logsource1
c     j5      : coefficients of shifted imag part logsource2
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer *4 nterms1,nterms2,ldc
      complex *16 h1(0:nterms1),h2(0:nterms1),h3(0:nterms1)
      complex *16 h4(0:nterms1),h5(0:nterms1)
      complex *16 h1a(0:1000),h2a(0:1000),h3a(0:1000)
      complex *16 h4a(0:1000),h5a(0:1000)
      complex *16 j1(0:nterms2),j2(0:nterms2),j3(0:nterms2)
      complex *16 j4(0:nterms2),j5(0:nterms2)
      complex *16 zc1,zc2,zdis,zdis1
      complex *16 zpow1(0:1000)
      complex *16 zzz,zzz1
      complex *16 zpow2(0:1000)
      real *8 carray(0:ldc,0:ldc)
      real *8 rscale1,rscale2,c1(2),c2(2)

      nmax=nterms1+nterms2
      
      zc1 = dcmplx(c1(1),c1(2))
      zc2 = dcmplx(c2(1),c2(2))
      zdis = 1/(zc1-zc2)
      zdis1 = (zc1-zc2)
      zpow1(0)=1.0d0
      zpow2(0)=1.0d0
      do i=1,nmax
           zpow1(i)=-zpow1(i-1)*zdis*rscale1
           zpow2(i)=zpow2(i-1)*zdis*rscale2
      enddo
      
      do i=0,nterms2
         j1(i)=0.0d0
         j2(i)=0.0d0
         j3(i)=0.0d0
         j4(i)=0.0d0
         j5(i)=0.0d0
      enddo

c     Handling the log term in the expansion
      j4(0)=h4(0)*log(cdabs(zdis1))
      j5(0)=h5(0)*log(cdabs(zdis1))
      do i=1,nterms2
          j4(i)=j4(i)-h4(0)/i
          j5(i)=j5(i)-h5(0)/i
      enddo
c
      do i=1,nterms1
          h1a(i)=h1(i)*zpow1(i)
          h2a(i)=(h2(i)-h3(i)*zdis1)*dconjg(zpow1(i))
          h3a(i)=h3(i)*dconjg(zpow1(i))
          h4a(i)=h4(i)*zpow1(i)
          h5a(i)=h5(i)*zpow1(i)
ccc          h2a(i)=h2a(i)-h3a(i)*zdis1
      enddo
      do i=0,nterms2
         do j=1,nterms1
             j1(i)=j1(i)+h1a(j)*carray(i+j-1,i)
             j2(i)=j2(i)+h2a(j)*carray(i+j-1,i)
             j3(i)=j3(i)+h3a(j)*carray(i+j-1,i)
             j4(i)=j4(i)+h4a(j)*carray(i+j-1,i)
             j5(i)=j5(i)+h5a(j)*carray(i+j-1,i)
        enddo
        j1(i)=j1(i)*zpow2(i)
        j2(i)=j2(i)*dconjg(zpow2(i))
        j3(i)=j3(i)*dconjg(zpow2(i))
        j4(i)=j4(i)*zpow2(i)
        j5(i)=j5(i)*zpow2(i)
      enddo

      return
      end
c*******************************************************************
