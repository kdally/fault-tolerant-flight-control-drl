"""
setup.py file for PH-LAB model
"""

import os
import numpy
import shutil
import glob
from distutils.core import setup, Extension

# Include directories
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
matlab_root = '/Applications/MATLAB_R2020a.app'
matlab_include = [
    matlab_root+'/extern/include',
    matlab_root+'/rtw/c/src', 
    matlab_root+'/rtw/c/src/ext_mode/common', 
    matlab_root+'/simulink/include',
    matlab_root+'/toolbox/simulink/simdemos/simfeatures/src'
    ]
include_dirs = [numpy_include] + matlab_include

source_files = ['swig_wrap.c',
                'c_citation.c', 
                'c_citation_data.c',
                'rt_look.c',
                'rt_look1d.c',
                'rt_look2d_normal.c',
                'rt_matrx.c',
                'rt_nonfinite.c',
                'rt_printf.c',
                'rtGetInf.c',
                'rtGetNaN.c',
                'ac_atmos.c',
                'ac_axes.c',
                'table3.c',
                'ert_main.c',
                ]

# Copy C files to local directory 
setup_dir = os.path.dirname(os.path.realpath(__file__))
for include_dir in include_dirs + [os.path.join(setup_dir, '..')]:
    for file_name in source_files:   
        if os.path.isfile(include_dir+'/'+file_name):
            print('Copying:', file_name)
            if not os.path.isfile(setup_dir+'/'+file_name):
                shutil.copy(include_dir+'/'+file_name, setup_dir)
            else:
                print('Aborted:', file_name, 'already exists in destination folder.')

# Create Module
cit_module = Extension( '_citation', 
                        sources=source_files, 
                        include_dirs=include_dirs,
                        define_macros=[
                            ('GENERATE_ASAP2', 0), 
                            ('EXTMODE_STATIC_ALLOC', 0), 
                            ('EXTMODE_STATIC_ALLOC_SIZE',1000000),
                            ('EXTMODE_TRANSPORT',0),
                            ('TMW_EXTMODE_TESTING',0)],
                        extra_compile_args=[
                            '-DNRT', 
                            '-DUSE_RTMODEL', 
                            '-DERT', 
                            '-DTID01EQ=1']
                        )

# Build module
setup ( name = 'citation', 
        version = '1.0', 
        ext_modules = [cit_module], 
        py_modules = ["citation"])

# Copy files
if os.path.isfile(setup_dir+'/citation.py'):
    shutil.copy(setup_dir+'/citation.py', os.path.join(setup_dir, '..'))

libs = [n for n in glob.glob('_citation*') if os.path.isfile(n)]
for lib in libs:
    shutil.copy(setup_dir+'/'+lib, os.path.join(setup_dir, '..'))
