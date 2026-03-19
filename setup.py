import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

include_dirs = [
    "src/waterz",
    "src/waterz/backend",
    numpy.get_include(),
]
if sys.platform == "darwin":
    include_dirs.append("/opt/homebrew/include")  # homebrew ... for boost

evaluate = Extension(
    name="waterz.evaluate",
    sources=[
        "src/waterz/evaluate.pyx",
        "src/waterz/frontend_evaluate.cpp",
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_link_args=["-std=c++11"],
    extra_compile_args=["-std=c++11", "-w"],
)

merge = Extension(
    name="waterz.merge",
    sources=[
        "src/waterz/merge.pyx",
        "src/waterz/frontend_merge.cpp",
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_link_args=["-std=c++11"],
    extra_compile_args=["-std=c++11", "-w", "-O2"],
)

region_graph = Extension(
    name="waterz.region_graph",
    sources=[
        "src/waterz/region_graph.pyx",
        "src/waterz/frontend_region_graph.cpp",
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_link_args=["-std=c++11"],
    extra_compile_args=["-std=c++11", "-w", "-O2"],
)

setup(ext_modules=cythonize([evaluate, merge, region_graph]))
