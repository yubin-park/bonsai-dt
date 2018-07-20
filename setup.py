from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "bonsai"
VERSION = "0.0.1"
DESCR = "Bonsai is a programmable decision tree framework."
URL = "https://yubin-park.github.io/bonsai-dt/"
REQUIRES = ["numpy", "cython"]

AUTHOR = "Yubin Park"
EMAIL = "yubin.park@gmail.com"

LICENSE = "Apache 2.0"

SRC_DIR = "bonsai"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".core._bonsai",
                  [SRC_DIR + "/core/_bonsai.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])
EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
