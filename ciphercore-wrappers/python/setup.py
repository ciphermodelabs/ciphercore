from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

__version__ = "0.3.0"

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
    rust_extensions=[RustExtension("ciphercore_internal", binding=Binding.PyO3, debug=False)],
    extras_require={"test": ["pytest", "numpy"], "dev": ["numpy"]},
    zip_safe=False,
    python_requires=">=3.7",
    package_dir={"": "py"},
    packages=find_packages(where="py"),
    include_package_data=True,
)
