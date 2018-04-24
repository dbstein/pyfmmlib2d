c    This file contains the common subroutines used across stokes
c    2d fmm codes
c
c
        subroutine bh2dzero(mpole,nterms)
c     This subroutine zeros out the first nterms in the
c     expansion nterms
c     
c     Input/output arguments
c     mpole    in/out: complex *16(0:nterms)
c              The expansion whose first nterms are to
c              be set to 0
c
c     nterms   in: integer
c              Number of terms in the expansion are to be
c              set to 0
c------------------------------------------------------------        
        implicit real *8 (a-h,o-z)
c
c       ... set multipole to zero
c
        complex *16 mpole(0:nterms)
c       
        do n=0,nterms
        mpole(n)=0
        enddo
c
        return
        end
c-------------------------------------------------------------        
        subroutine bh2dadd(mpole,mpole2,nterms)
c      This subroutine adds the expansion in mpole to mpole2
c      Both of the expansions are assumed to be of the same
c      length
c
c      Input arguments
c      mpole:   in:complex *16 (0:nterms)
c               multipole expansion whose contribution
c               is to be added to mpole2
c
c     mpole2    in/out:complex *16 (0:nterms)
c               multipole expansion to which the contribution
c               of mpole is to be added. On output, the
c               updated expansion is returned
c
c    nterms     in: Integer
c               number of terms in the expansion
c-------------------------------------------------------------
        implicit real *8 (a-h,o-z)
        complex *16 mpole(0:nterms)
        complex *16 mpole2(0:nterms)
c       
        do n=0,nterms
        mpole2(n)=mpole2(n)+mpole(n)
        enddo
c
        return
        end
c---------------------------------------------------------        
        subroutine bh2dadd2(mpole,ntbig,mpole2,nterms)
c      This subroutine adds the expansion in mpole to mpole2
c      Both of the expansions need not be of the same
c      length
c
c      Input arguments
c      mpole:   in:complex *16 (0:ntbig)
c               multipole expansion whose contribution
c               is to be added to mpole2
c
c     ntbig     in:integer
c               number of terms in the mpole expansion
c
c     mpole2    in/out:complex *16 (0:nterms)
c               multipole expansion to which the contribution
c               of mpole is to be added. On output, the
c               updated expansion is returned
c
c    nterms     in: Integer
c               number of terms of the expansion of mpole which
c               need to be added
c
c    Note: ntbig must be greater than nterms
c----------------------------------------------------------
        implicit real *8 (a-h,o-z)
        complex *16 mpole(0:ntbig)
        complex *16 mpole2(0:nterms)
c       
        do n=0,nterms
           mpole2(n)=mpole2(n)+mpole(n)
        enddo
c
        return 
        end
c----------------------------------------------------------       
        subroutine bh2dmpalloc(wlists,iaddr,nboxes,lmptot,nterms)
c     This subroutine determines the size of the array
c     to be allocated for the multipole expansions
c     iaddr(1,i) points to the starting location of the multipole
c     expansion of box i, iaddr(2,i) points to
c     z times conjugate multipole expansion, iaddr(3,i) points
c     to the conjugate multipole expansion, iaddr(4,i) points
c     to the multipole expansion corresponding to the real
c     part of the log charge density and iaddr(5,i) points
c     to the multipole expansion corresponding to the imaginary
c     part of the log charge density. iaddr(6:10,i) are the
c     corresponding local expansions
c  
c     Input arguments
c     wlists      in: real *8
c                 Lists for the given adaptive quad tree
c
c     nboxes      in: integer
c                 total number of boxes in the tree
c     
c     nterms      in: Integer(0:nlevels)
c                 Number of terms requried in expansions at each
c                 level
c
c------------------------------------------------------------------
c     Output arguments
c     iaddr       out: Integer(10,nboxes)
c                 Points the multipole and local expansions in box i
c 
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------
        implicit real *8 (a-h,o-z)
        integer box(20)
        integer nterms(0:*)
        integer iaddr(10,nboxes)
        real *8 center0(2),corners0(2,4)
        real *8 wlists(*)
c
c       ... construct pointer array iaddr for addressing multipole and
c       local expansion
c
        iptr=1
        do ibox=1,nboxes
        call d2tgetb(ier,ibox,box,center0,corners0,wlists)
        level=box(1)
c
c       ... first, allocate memory for the multipole expansion
c       
        iaddr(1,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(2,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(3,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(4,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(5,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2
c       ... then, allocate memory for the local expansion
c       
        iaddr(6,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(7,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(8,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(9,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2

        iaddr(10,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*2
c       
        enddo
        lmptot = iptr
c        print *, 'lmptot=',lmptot
        return
        end
c-------------------------------------------------------------        
        subroutine bh2d_init_carray(carray,ldc)
c------------------------------------------------------------
c     This subroutine computes the matrix of binomial
c     coeffients where the i,j th entry for i>j denotes i choose
c     j
c
c     input arguments
c     ldc    in: integer
c            size of the matrix of binomial coefficients
c            to be computed
c
c     output arguments
c     carray    real *8 (0:ldc, 0:ldc)
c        
        implicit real *8 (a-h,o-z)
        real *8 carray(0:ldc,0:ldc)

        do l = 0,ldc
        carray(l,0) = 1.0d0
        enddo
        do m=1,ldc
        carray(m,m) = 1.0d0
        do l=m+1,ldc
            carray(l,m)=carray(l-1,m)+carray(l-1,m-1)
        enddo
        enddo
c
        return
        end
     
c--------------------------------------------------------------        
      subroutine bh2dtexpsort(n,isource,ntj,t1sort,t1,
     1   t2sort,t2,t3sort,t3,t4sort,t4,t5sort,t5,scjsort,scj)
c------------------------------------------------------------
c     This subroutine sorts into original ordering the complex
c     arrays of expansions t1sort,t2sort,t3sort,t4sort,t5sort
c     and real array scjsort into the corresponding arrays
c     t1,t2,t3,t4,t5 and scj where isource defines the mapping
c     from the unordered array to the tree sorted array
c     Input arguments
c     n      in: Integer
c            number of sources
c
c     isource   in:Integer(n)
c             tree sorted ordering of sources
c
c     ntj     in: Integer
c             number of terms in th expansion
c
c     Treesorted arrays
c     t1sort, t2sort, t3sort, t4sort, t5sort
c     Dimension (0:ntj,1:n)
c     scjsort(1:n)
c
c     OUTPUT
c     Arrays in the user defined ordering
c     t1,t2,t3,t4,t5
c     Dimension (0:ntj,1:n)
c
c     scj(1:n)
c------------------------------------------------------------
       implicit real *8 (a-h,o-z)
       integer isource(*)
       complex *16 t1sort(0:ntj,*),t1(0:ntj,*)
       complex *16 t2sort(0:ntj,*),t2(0:ntj,*)
       complex *16 t3sort(0:ntj,*),t3(0:ntj,*)
       complex *16 t4sort(0:ntj,*),t4(0:ntj,*)
       complex *16 t5sort(0:ntj,*),t5(0:ntj,*)
       real *8 scjsort(*),scj(*)
c        
        do i=1,n
           scj(isource(i))=scjsort(i)
           do jj=0,ntj
              t1(jj,isource(i))=t1sort(jj,i)
              t2(jj,isource(i))=t2sort(jj,i)
              t3(jj,isource(i))=t3sort(jj,i)
              t4(jj,isource(i))=t4sort(jj,i)
              t5(jj,isource(i))=t5sort(jj,i)
           enddo
        enddo
c
      return
      end
c-------------------------------------------------------------      
        subroutine bh2dreorder(nsource,source,
     $  ifcharge,charge,isource,ifdipole,
     1  dip1,dip2,sourcesort,chargesort,dip1sort,dip2sort)
c--------------------------------------------------------------
c       This subroutine tree orders the sources, charges and
c       dipoles into thee respective sorted arrays sourcesort,
c       chargesort, dip1sort and dip2sort.

c       Input arguments
c       nsource         in: Integer
c                       number of sources
c
c       source          in: real *8 (2,nsource)
c                       unsorted array of sources
c 
c       ifcharge        in: Integer
c                       flag for sorting charges. If ifcharge=1,
c                       then the array of charges will be tree
c                       sorted
c      
c       isource        in: integer
c                      mapping from unsorted array to tree sorted
c                      array, isource(i) in the unsorted array
c                      maps to the ith element in the tree sorted
c                      array
c
c       charge         in: complex *16
c                       unsorted array of charges
c
c       dip1            in: complex *16
c                       unsorted dipole1 array
c
c       dip2            in: complex *16
c                       unsorted dipole2 array
c
c-------------------------------------------------------------
c      OUTPUT
c      sourcesort     out: real *8
c                     sorted array of sources
c 
c      chargesort     out: complex *16
c                     sorted array of charges
c
c      dip1sort       out: complex *16
c                     sorted array of dipole1
c
c      dip2sort       out: complex *16
c                     sorted array of dipole2
c--------------------------------------------------------------
        implicit none
        real *8 source(2,*),sourcesort(2,*)
        integer isource(*),i,nsource

        integer ifcharge,ifdipole
        complex *16 charge(*),chargesort(*),dip1(*),dip1sort(*)
        complex *16 dip2(*),dip2sort(*)


C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
        do i = 1,nsource
           sourcesort(1,i) = source(1,isource(i))
           sourcesort(2,i) = source(2,isource(i))
           if( ifcharge .ge. 1 ) then
              chargesort(i) = charge(isource(i))
           endif
           if (ifdipole .ge. 1) then
              dip1sort(i) = dip1(isource(i))
              dip2sort(i) = dip2(isource(i))
           endif
        enddo
C$OMP END PARALLEL DO
        return
        end
c
c-----------------------------------------------------------------
       
      subroutine bh2dreordertarg(ntarget,targ,itarget,
     1                                  targsort)  
c     This subroutine tree sorts the targets into the targsort
c     array
c     Input arguments
c     ntarget       in: Integer
c                   number of targets
c
c     target        in: real*8 (2,ntarget)
c                   unsorted array of targets
c
c     itarget       in: integer(ntarget)
c                   mapping for tree sorting the targets
c
c ----------------------------------------------------------------
c     Output arguments
c     targsort      in: real *8 (2,ntarget)
c                   tree ordered array of targets
c---------------------------------------------------------------
      implicit none
      integer i,itarget(*),ntarget
      real *8 targ(2,1),targsort(2,1)

      do i=1,ntarget
         targsort(1,i) = targ(1,itarget(i))
         targsort(2,i) = targ(2,itarget(i))
      enddo

      return
      end
c-----------------------------------------------------------------      
      subroutine bh2dmpalloc_newtree(laddr,iaddr,nlevels,lmptot,
     1                               nterms)  
c     This subroutine determines the size of the array
c     to be allocated for the multipole expansions
c     iaddr(1,i) points to the starting location of the multipole
c     expansion of box i, iaddr(2,i) points to
c     z times conjugate multipole expansion, iaddr(3,i) points
c     to the conjugate multipole expansion, iaddr(4,i) points
c     to the multipole expansion corresponding to the real
c     part of the log charge density and iaddr(5,i) points
c     to the multipole expansion corresponding to the imaginary
c     part of the log charge density. iaddr(6:10,i) are the
c     corresponding local expansions
c  
c     Input arguments
c     laddr       in: Integer(2,0:nlevels)
c                 indexing array provinding access to boxes at each
c                 level
c
c     nlevels     in: Integer
c                 Total numner of levels
c     
c     nterms      in: Integer(0:nlevels)
c                 Number of terms requried in expansions at each
c                 level
c
c------------------------------------------------------------------
c     Output arguments
c     iaddr       out: Integer(10,nboxes)
c                 Points the multipole and local expansions in box i
c 
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------
      implicit none
      integer laddr(2,0:nlevels),nlevels,nterms(0:nlevels)
      integer iaddr(10,1), lmptot
      integer ibox,i,iptr

      iptr = 1
      lmptot = 0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c            Allocate memory for the multipole expansions
             iaddr(1,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(2,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(3,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(4,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(5,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

c            Allocate memory for the local expansion
             iaddr(6,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(7,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(8,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(9,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2

             iaddr(10,ibox) = iptr
             iptr = iptr + (nterms(i)+1)*2
         enddo
      enddo
      lmptot = iptr

      return
      end
c-----------------------------------------------------------------      
      subroutine bh2dpsort(ns,imap,arrsort,arr)
c     This subroutine sorts into original ordering the complex
c     array arrsort to arr via the mapping imap
c
c     Input arguments
c     ns         in: Integer
c                number of elements to be unsorted
c     
c     imap       in: Integer (ns)
c                mapping used to sort the arrays
c
c     arrsort    in: complex *16 (ns)
c                sorted array of velocity/gradients
c---------------------------------------------------------------
c     Output arguments
c     arr        out: complex *16 (ns)
c                resorted array into original ordering
c---------------------------------------------------------------
      implicit none
      integer ns,i,imap(1)
      complex *16 arrsort(1),arr(1)

      do i=1,ns
         arr(imap(i)) = arrsort(i)
      enddo

      return
      end
c--------------------------------------------------------------      

      subroutine bh2dreorderint(n,arr,isort,arrsort)
c    This subroutine reorders the array of integers
c    to the array arrsort via the mapping isort.
c    The ordering is reset to arrsort(i) = arr(isort(i))
c
c    INPUT arguments
c    n       in:Integer
c            Number of elements in the array
c
c    arr     in:Integer(n)
c            The array to be sorted
c
c    isort   in:Integer(n)
c            Mapping used to sort the array
c
c    OUTPUT arguments
c    arrsort  in:Integer(n)
c             The sorted array
c -------------------------------------------------------
      implicit none
      integer n,i,arr(1),arrsort(1),isort(1)

      do i=1,n
         arrsort(i) = arr(isort(i))
      enddo

      return
      end
c----------------------------------------------------------      
