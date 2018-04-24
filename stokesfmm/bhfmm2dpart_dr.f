c
c     testing code for FMM - tests charges and dipoles against
c     O(N^2) direct method 
c
c
        implicit real *8 (a-h,o-z)
        real *8 source(2,2 000 000)
        complex *16 charge(2 000 000)
        complex *16 dip1(2 000 000)
        complex *16 dip2(2 000 000)
        complex *16 vel(2 000 000)
        complex *16 grada(2 000 000)
        complex *16 gradaa(2 000 000)
c       
        complex *16 vel2(2 000 000)
        complex *16 grada2(2 000 000)
        complex *16 gradaa2(2 000 000)
c       
        real *8 target(2,2 000 000)
        complex *16 veltarg(2 000 000)
        complex *16 gradatarg(2 000 000)
        complex *16 gradaatarg(2 000 000)
c
        complex *16 vtemp,gatemp,gaatemp
c       
        complex *16 ima
        data ima/(0.0d0,1.0d0)/
c
        done=1
        pi=4*atan(done)
c
c       SET ALL PARAMETERS
c        
        call prini(6,13)

        print *, 'ENTER n'
        read *, nsource
  
        call prinf('ierr=*',ierr,1)
c
        call prinf('nsource=*',nsource,1)
c
c       ... construct randomly located charge distribution on a unit circle
        done=1
        pi=4*atan(done)
c
        d=hkrand(0)
        do i=1,nsource
           phi=hkrand(0)*2*pi
           source(1,i)=cos(phi)
           source(2,i)=sin(phi)
        enddo
c
c       ... construct target distribution on a unit circle
c
        ntarget=nsource*1
        do i=1,ntarget
           phi=hkrand(0)*2*pi
           target(1,i)=cos(phi)+3.0d0
           target(2,i)=sin(phi)
        enddo
c
        call prinf('ntarget=*',ntarget,1)
c
        scale=1
        do i=1,nsource
           source(1,i)=source(1,i)*scale
           source(2,i)=source(2,i)*scale
        enddo
        do i=1,ntarget
           target(1,i)=target(1,i)*scale
           target(2,i)=target(2,i)*scale
        enddo

        iprec=4
        call prinf('iprec=*',iprec,1)
c       
        ifvel=1
        ifgrada=1
c
        ifcharge=1
        ifdipole=1
c
        ifveltarg=1
        ifgradatarg=1
c
        ifgradaa=1
        ifgradaatarg=1
c
        if( ifveltarg .eq. 0 .and. ifgradatarg .eq. 0 
     $     .and. ifgradaatarg .eq. 0 ) ntarget = 0
c
        if (ifcharge .eq. 1 ) then
           do i=1,nsource
              charge(i)=1.0d0/i
           enddo
        endif
c       
        if (ifdipole .eq. 1) then
           do i=1,nsource
              dip1(i)=1/i+ima/(2*i)
              dip2(i)=1/i-ima/(2*i)
           enddo
        endif
c
        t1=second()
C$        t1=omp_get_wtime()
c       
        call bhfmm2dparttarg(ier,iprec,
     $     nsource,source,
     $     ifcharge,charge,ifdipole,dip1,dip2,
     $     ifvel,vel,ifgrada,grada,ifgradaa,gradaa,
     $     ntarget,target,ifveltarg,veltarg,ifgradatarg,gradatarg,
     $     ifgradaatarg,gradaatarg)
        t2=second()
C$        t2=omp_get_wtime()

c       
c       
        call prinf('nsource=*',nsource,1)
        call prinf('ntarget=*',ntarget,1)
        call prin2('after fmm, time (sec)=*',t2-t1,1)
        call prin2('after fmm, speed (points+targets/sec)=*',
     $     (nsource+ntarget)/(t2-t1),1)
        
        m=min(nsource,12)
c
        ifprint=0
        if (ifprint .eq. 1) then
           call prin2('source=*',source,3*nsource)
        endif
c
        do i=1,nsource
           if (ifvel.eq.1) vel2(i)=0
           if (ifgrada.eq.1) then
              grada2(i)=0
           endif
           if (ifgradaa.eq. 1) then
              gradaa2(i)=0
           endif
        enddo
        t1=second()
C$      t1=omp_get_wtime() 
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,vtemp,gatemp,gaatemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do 7160 j=1,m
           do 7150 i=1,nsource       
              if( i .eq. j ) goto 7150
              call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     1        ifdipole,dip1(i),dip2(i),source(1,j),vtemp,ifgrada,
     2        gatemp,ifgradaa,gaatemp)
              if (ifvel .eq. 1) vel2(j)=vel2(j)+vtemp
              if (ifgrada .eq. 1) then
                 grada2(j)=grada2(j)+gatemp
              endif
              if (ifgradaa .eq. 1) then
                 gradaa2(j)=gradaa2(j)+gaatemp
              endif
 7150      continue
 7160   continue
C$OMP END PARALLEL DO
        t2=second()
C$      t2=omp_get_wtime() 
        ifprint = 1
        if (ifprint .eq. 1) then
           if( ifvel.eq.1 ) then
              call prin2('fmm, vel=*',vel,2*m)
              call prin2('directly, vel=*',vel2,2*m)
           endif

           if( ifgrada.eq.1 ) then
              call prin2('fmm, grada=*',grada,2*m)
              call prin2('directly, grada=*',grada2,2*m)
           endif

           if( ifgradaa.eq.1 ) then
              call prin2('fmm, gradaa=*',gradaa,2*m)
              call prin2('directly, gradaa=*',gradaa2,2*m)
           endif
        endif
c
        call prin2('directly, estimated time (sec)=*',
     $     (t2-t1)*dble(nsource)/dble(m),1)
        call prin2('directly, estimated speed (points/sec)=*',
     $     m/(t2-t1),1)
c
c        open(unit=1,file='output.dat',access='append')
c        write(1,*) "ns =",nsource

        if (ifvel .eq. 1)  then
           call l2derror(vel,vel2,m,aerr,rerr)
           call prin2('relative L2 error in potential=*',rerr,1)
c           write(1,*) "err vel =",rerr
           call l2derror2(vel,vel2,m,aerr,rerr)
           call prin2('relative L2 error in re(potential)=*',rerr,1)
c           write(1,*) "err Re(vel)=",rerr
        endif
c
        if (ifgrada.eq. 1) then
           call l2derror(grada,grada2,m,aerr,rerr)
           call prin2('relative L2 error in gradient=*',rerr,1)
c           write(1,*) "err grada=",rerr
        endif
c       
        if (ifgradaa.eq. 1) then
           call l2derror(gradaa,gradaa2,m,aerr,rerr)
           call prin2('relative L2 error in hessian=*',rerr,1)
c           write(1,*) "err gradaa=",rerr
        endif
c       
c
        if( ntarget .eq. 0 ) stop
c
        do i=1,ntarget
           if (ifveltarg .eq. 1) vel2(i)=0
           if (ifgradatarg .eq. 1) then
              grada2(i)=0
           endif
           if (ifgradaatarg .eq. 1) then
              gradaa2(i)=0
           endif
        enddo
        t1=second()
C$      t1=omp_get_wtime() 
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,vtemp,gatemp,gaatemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do 8160 j=1,m
           do 8150 i=1,nsource
              call bh2ddireval(source(1,i),1,ifcharge,charge(i),
     1        ifdipole,dip1(i),dip2(i),target(1,j),vtemp,
     2        ifgradatarg,gatemp,ifgradaatarg,gaatemp)        
              if (ifveltarg .eq. 1) vel2(j)=vel2(j)+vtemp
              if (ifgradatarg .eq. 1) then
                 grada2(j)=grada2(j)+gatemp
              endif
              if (ifgradaatarg .eq. 1) then
                 gradaa2(j)=gradaa2(j)+gaatemp
              endif
 8150      continue
 8160   continue
C$OMP END PARALLEL DO
        t2=second()
C$      t2=omp_get_wtime() 
c
        if (ifprint .eq. 1) then
           if (ifveltarg .eq. 1) 
     $        call prin2('after fmm, veltarg=*',veltarg,2*m)
           if (ifveltarg .eq. 1) 
     $        call prin2('directly, veltarg=*',vel2,2*m)
           if( ifgradatarg.eq.1 ) 
     $        call prin2('after fmm, gradatarg=*',gradatarg,2*m)
           if( ifgradatarg.eq.1 ) 
     $        call prin2('directly, gradatarg=*',grada2,2*m)
           if( ifgradaatarg.eq.1 ) 
     $        call prin2('after fmm, gradaatarg=*',gradaatarg,2*m)
           if( ifgradaatarg.eq.1 ) 
     $        call prin2('directly, gradaatarg=*',gradaa2,2*m)
        endif
c
        call prin2('directly, estimated time (sec)=*',
     $     (t2-t1)*dble(ntarget)/dble(m),1)
        call prin2('directly, estimated speed (targets/sec)=*',
     $     m/(t2-t1),1)
c       
        if (ifveltarg .eq. 1) then
           call l2derror(veltarg,vel2,m,aerr,rerr)
           call prin2('relative L2 error in target velocity=*',rerr,1)
c           write(1,*) "err vel targ=",rerr
           call l2derror2(veltarg,vel2,m,aerr,rerr)
           call prin2
     $     ('relative L2 error in target re(velocity)=*',rerr,1)
c           write(1,*) "err Re(vel) targ=",rerr
        endif
c
        if (ifgradatarg .eq. 1) then
           call l2derror(gradatarg,grada2,m,aerr,rerr)
           call prin2('relative L2 error in target gradient=*',rerr,1)
c           write(1,*) "err grada targ=",rerr
        endif
c       
        if (ifgradaatarg .eq. 1) then
           call l2derror(gradaatarg,gradaa2,m,aerr,rerr)
           call prin2('relative L2 error in target hessian=*',rerr,1)
c           write(1,*) "err gradaa targ=",rerr
        endif
c        write(1,*) ""
c        write(1,*) ""
c        close(1)
c       
        stop
        end
c
c
c
        subroutine l2dmperr(mpole1,mpole2,nterms,d)
        implicit real *8 (a-h,o-z)
c       
        complex *16 mpole1(-nterms:nterms)
        complex *16 mpole2(-nterms:nterms)
c       
        d=0
c       
        do n=-nterms,nterms
        d=d+abs(mpole1(n)-mpole2(n))**2
        enddo
c       
        d=d/(2*nterms+1)
        d=sqrt(d)
c       
        return
        end
c
c
c
c
c
        subroutine l2dmpnorm(mpole,nterms,d)
        implicit real *8 (a-h,o-z)
c
        complex *16 mpole(-nterms:nterms)
c
        d=0
c
        do n=-nterms,nterms
        d=d+abs(mpole(n))**2
        enddo
c
        d=d/(2*nterms+1)
        d=sqrt(d)
c
        return
        end
c
c
c
c
c
        subroutine l2derror(pot1,pot2,n,ae,re)
        implicit real *8 (a-h,o-z)
c
c       evaluate absolute and relative errors
c
        complex *16 pot1(n),pot2(n)
c
        d=0
        a=0
c       
        do i=1,n
        d=d+abs(pot1(i)-pot2(i))**2
        a=a+abs(pot1(i))**2
        enddo
c       
        d=d/n
        d=sqrt(d)
        a=a/n
        a=sqrt(a)
c       
        ae=d
        re=d/a
c       
        return
        end
c
c
c
c
c
        subroutine l2derror2(pot1,pot2,n,ae,re)
        implicit real *8 (a-h,o-z)
c
c       evaluate absolute and relative errors of the real part
c
        complex *16 pot1(n),pot2(n)
c
        d=0
        a=0
c       
        do i=1,n
        d=d+dble(pot1(i)-pot2(i))**2
        a=a+dble(pot1(i))**2
        enddo
c       
        d=d/n
        d=sqrt(d)
        a=a/n
        a=sqrt(a)
c       
        ae=d
        re=d/a
c       
        return
        end
c
c
c
