from setuptools import setup, find_packages
from codecs import open
from os import path
from numpy.distutils.core import setup, Extension

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyfmmlib2d',
    version='0.0.1',
    description='Python Wrapper for 2D Laplace/Helmholtz/Stokes FMMs',
    long_description=long_description,
    url='https://github.com/dbstein/pyfmmlib2d',
    author='David Stein',
    author_email='dstein@flatironinstitute.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists/Mathematicians',
        'License :: Apache 2',
        'Programming Language :: Python :: 2',
    ],
    ext_modules = [
                    Extension( 'stokesfmm',
                        ['stokesfmm/bhfmm2dpart.f',
                         'stokesfmm/bhcommon_oldtree.f',
                         'stokesfmm/bhfmm2dpart_dr.f',
                         'stokesfmm/d2tstrcr_omp.f',
                         'stokesfmm/bhfmm2drouts.f',
                         'stokesfmm/bhrouts.f',
                         'stokesfmm/d2mtreeplot.f',
                         'stokesfmm/dlaran.f',
                         'stokesfmm/hkrand.f',
                         'stokesfmm/l2dterms.f',
                         'stokesfmm/laprouts2d.f',
                         'stokesfmm/prini.f',
                        ],
                        extra_f77_compile_args=['-O3','-fopenmp'],
                        extra_f90_compile_args=['-O3','-fopenmp'],
                        libraries=['iomp5',],),
                        # note the linkage to iomp5, if you link to gomp instead
                        # this breaks threaded MKL linkage in numpy on MACOSX
                    Extension( 'fmmlib2d',
                        ['fmmlib2d/src/cdjseval2d.f',
                         'fmmlib2d/src/cfmm2dpart.f',
                         'fmmlib2d/src/d2mtreeplot.f',
                         'fmmlib2d/src/d2tstrcr_omp.f',
                         'fmmlib2d/src/h2dterms.f',
                         'fmmlib2d/src/hank103.f',
                         'fmmlib2d/src/helmrouts2d.f',
                         'fmmlib2d/src/hfmm2dpart.f',
                         'fmmlib2d/src/hfmm2drouts.f',
                         'fmmlib2d/src/l2dterms.f',
                         'fmmlib2d/src/laprouts2d.f',
                         'fmmlib2d/src/lfmm2dpart.f',
                         'fmmlib2d/src/lfmm2drouts.f',
                         'fmmlib2d/src/prini.f',
                         'fmmlib2d/src/rfmm2dpart.f',
                         'fmmlib2d/src/second-r8.f',
                         'fmmlib2d/src/zfmm2dpart.f',
                        ],
                        extra_f77_compile_args=['-O3','-fopenmp'],
                        extra_f90_compile_args=['-O3','-fopenmp'],
                        libraries=['iomp5',],)
                    ],
    packages=find_packages(),
    install_requires=[],
)
