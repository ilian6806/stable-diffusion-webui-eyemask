import os
import sys

from launch import is_installed, run, git_clone
from modules.paths import models_path
from modules.sd_models import model_hash
from modules import modelloader
from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")

def list_models(model_path):
    model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])

    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    models = []
    for filename in model_list:
        h = model_hash(filename)
        title, short_model_name = modeltitle(filename, h)
        models.append(title)

    return models

print('Installing requirements for eyemask')

if not is_installed("dlib"):
    python = sys.executable
    run(f'"{python}" -m pip install setuptools', desc="Installing setuptools", errdesc="Couldn't install setuptools")
    run(f'"{python}" -m pip install dlib', desc="Installing dlib", errdesc="Couldn't install dlib")

if not is_installed("mmdet"):
    python = sys.executable
    run(f'"{python}" -m pip install -U openmim', desc="Installing openmim", errdesc="Couldn't install openmim")
    run(f'"{python}" -m mim install mmcv-full', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
    run(f'"{python}" -m pip install mmdet', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

if (len(list_models(dd_models_path)) == 0):
    print("No detection models found, downloading...")
    bbox_path = os.path.join(dd_models_path, "bbox")
    segm_path = os.path.join(dd_models_path, "segm")
    load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
    load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)
    load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)
    load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/segm/mmdet_dd-person_mask2former.py", segm_path)

git_clone("https://github.com/isl-org/MiDaS.git", "repositories/midas", "midas")
