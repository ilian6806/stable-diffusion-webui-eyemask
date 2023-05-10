'''
Eye Mask - Stable Diffusion Web UI extension for mark and redraw eyes/faces.
Core logic for 'run' method is in eyemask.script.

Author: ilian.iliev
Since: 09.01.2023
'''

import os
import sys

from modules import scripts
sys.path.append(os.path.join(scripts.basedir(), 'scripts'))

from eyemask import constants, ui, script as eye_mask_script


class EyeMaskScript(scripts.Script):

    def __init__(self, *k, **kw):
        self.eye_mask_core = eye_mask_script.EyeMasksCore()
        self.eye_mask_ui = ui.EyeMaskUI(self)
        super().__init__()

    def title(self):
        return constants.script_name

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        return self.eye_mask_ui.render(is_img2img)

    def run(self, *args, **kwargs):
        return self.eye_mask_core.execute(*args, **kwargs)
