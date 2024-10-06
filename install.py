import os
import sys

from launch import is_installed, run, git_clone
from modules.paths import models_path
from modules import modelloader
from basicsr.utils.download_util import load_file_from_url

python = sys.executable

if not is_installed("cmake"):
    run(f'"{python}" -m pip install cmake', desc="Installing cmake", errdesc="Couldn't install cmake")

if not is_installed("setuptools"):
    run(f'"{python}" -m pip install setuptools', desc="Installing setuptools", errdesc="Couldn't install setuptools")

if not is_installed("dlib"):
    try:
        run(f'"{python}" -m pip install dlib==19.24.0', desc="Installing dlib", errdesc="Couldn't install dlib")
    except Exception as e:
        print(e)
        print("----------------------------------------------")
        print("Failed building wheel for dlib")
        print("ERROR: CMake must be installed to build dlib")
        print("Install cmake from https://cmake.org/download/")
        print("----------------------------------------------")

git_clone("https://github.com/isl-org/MiDaS.git", "repositories/midas", "midas")
