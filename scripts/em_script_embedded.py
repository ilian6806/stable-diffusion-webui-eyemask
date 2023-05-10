'''
Eye Mask - Stable Diffusion Web UI extension (embedded version) for mark and redraw eyes/faces.
Core logic is in eyemask.script_embedded.

Author: ilian.iliev
Since: 09.01.2023
'''

import os
import sys

from modules import scripts
sys.path.append(os.path.join(scripts.basedir(), 'scripts'))

from eyemask import constants, ui, script_embedded as eye_mask_script
import modules.shared as shared


class EyeMaskEmbeddedScript(scripts.Script):

    def __init__(self, *k, **kw):
        self.eye_mask_core = eye_mask_script.EyeMasksEmbeddedCore()
        self.eye_mask_ui = ui.EyeMaskUI(self)
        super().__init__()

    def title(self):
        return constants.script_name

    def show(self, is_img2img):
        try:
            return scripts.AlwaysVisible if shared.opts.em_show_embedded_version else False
        except Exception as e:
            return False

    def ui(self, is_img2img):
        return self.eye_mask_ui.render(is_img2img)

    def process(self, p, *args):
        return self.eye_mask_core.execute_process(p, *args)

    def postprocess(self, p, processed, *args):
        return self.eye_mask_core.execute_postprocess(p, processed, *args)
