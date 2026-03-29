# setup.py
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="macd",
        sources=["macd.pyx"],
        include_dirs=[numpy.get_include()],
        language="c"
    )
]

setup(
    name="macd",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
