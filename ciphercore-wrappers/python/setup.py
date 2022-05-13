
import sys

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
import setuptools
from setuptools import setup

__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension("ciphercore_native",
        ["src/ciphercore_native.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=["include/"],
        library_dirs=["lib/"],
        libraries=["cadapter", "ssl", "crypto"],
        ),
]

with open('../../README.md', 'r') as f:
  long_description = f.read()

setup(
    name="ciphercore",
    version=__version__,
    author="CipherMode",
    author_email="info@ciphermode.com",
    url="https://github.com/ciphermodelabs/ciphercore/",
    description="Python wrapper for CipherCore base library (graph building part)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="Apache 2.0",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    package_dir={"": "py"},
    packages=setuptools.find_packages(where="py"),
    include_package_data=True,
)
