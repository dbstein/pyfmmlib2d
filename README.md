# pyfmmlib2d: Laplace, Helmholtz, and Stokes Fast Multipole Methods in 2D
 ---
A pythonic wrapper for two Fast Multipole Method libraries:
1. FMMLIB2D, the Fast Multipole Method library for Laplace and Helmholtz equations in two dimensions, written by Leslie Greengard and Zydrunas Gimbutas, and available at: https://github.com/zgimbutas/fmmlib2d
2. A 2D Stokes Library, written by Manas Rachh (link?)

A unified wrapper with some amount of documentation for the high level FMM functions of both packages is available through the FMM function. Undocumented low level interfaces to all functions are available through the \fmmlib2d and \stokesfmm submodules. A basic test suite is included that can be run with pytest (if you have pytest installed, just type "pytest" in the terminal when in the same directory where the install.py file is located).

### Installation:

Simply navigate to root of the directory and type:

```bash
pip install .
```

I believe all you really need for the setup to run is a numpy, gfortran, and openmp. Some caveats:
1. I have by default linked to the Intel OpenMP. If you are using Mac OSX, be careful with this! If you are using a numpy/scipy distribution that is threaded, this should link to the same OpenMP library, otherwise you may get erratic/wrong behavior (in either numpy/scipy or this package, depending on the order in which you load them).
2. The installer will not check that you have numpy installed (in particular, this is to avoid breaking peoples conda-based installations by having pip update things).
3. If you don't have OpenMP installed, just remove lines 45/46 from setup.py.
4. If you have a different Fortran compiler than gfortran installed you will have to make the other obvious modifications to setup.py (compile and link flags). I have compiled this with both gfortran and ifort without issue.

### Testing
I have written and included some basic functionality tests that can be run with pytest (if you have pytest installed, just type "pytest" in the terminal when in the same directory where the install.py file is located). I have not written any tests for the Helmholtz and biharmonic FMMs. Tests exist for all other FMMs, including the Stokes wrapper to the biharmonic FMM.

### Known Problems
I attempted briefly to write a test program for the Helmholtz FMM. The HFMM wrapper throws an error, because the hfmm2dparttarg subroutine wants arrays of size (k,1) instead of size (k, n_target), which is what I was expecting.  Because I don't currently need Helmholtz FMMs, I've left looking into this to a later date.
