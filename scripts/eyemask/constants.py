import os
from modules import scripts

script_name = 'Eye Mask'

script_base_dir = scripts.basedir()
script_static_dir = os.path.join(script_base_dir, 'static')
script_models_dir = os.path.join(script_base_dir, 'models')
script_wildcards_dir = os.path.join(script_base_dir, 'wildcards')