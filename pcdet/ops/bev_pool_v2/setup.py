from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import glob

src_files = glob.glob('src/*.cpp') + glob.glob('src/*.cu')
print(src_files)

setup(
    name='bevpoolv2',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_v2_ext',
            sources=src_files,
            # include_dirs=[' '],
            # libraries=['cuhash'],
            # library_dirs=['point_sample/cuhash/'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})

