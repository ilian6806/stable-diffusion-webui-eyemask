import datetime
import importlib
import gradio as gr

import modules.sd_models
from modules.ui_components import ToolButton

from . import constants, script, script_embedded, utils, widlcards, state, mask_generator
from modules import shared


class EyeMaskUI():

    def __init__(self, eye_mask_script):
        self.eye_mask_script = eye_mask_script
        self.eye_mask_core = eye_mask_script.eye_mask_core
        self.is_embedded = isinstance(self.eye_mask_core, script_embedded.EyeMasksEmbeddedCore)

    def restart(self):
        # interrupt current image processing
        shared.state.interrupt()
        # reimport all dynamic packs
        importlib.reload(utils)
        importlib.reload(state)
        importlib.reload(widlcards)
        importlib.reload(mask_generator)
        # instantiate again core logic
        if self.is_embedded:
            importlib.reload(script_embedded)
            self.eye_mask_script.eye_mask_core = script_embedded.EyeMasksEmbeddedCore()
        else:
            importlib.reload(script)
            self.eye_mask_script.eye_mask_core = script.EyeMasksCore()
        # update UI
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{constants.script_name} reloaded. Last reload {now}')
        return f'<div style="text-align:center; margin-bottom: 8px;">Last reload: {now}</div>'

    def get_elem_id_prefix(self):
        result = 'em-'
        if self.is_embedded:
            result += 'emb-'
        return result

    def get_id(self, id, is_img2img):
        return "%s%s-%s" % (self.get_elem_id_prefix(), id, "img2img" if is_img2img else "txt2img")

    def render(self, is_img2img):
        if self.is_embedded:
            with gr.Group():
                with gr.Accordion(constants.script_name, open=False):
                    with gr.Group(elem_id=self.get_id("eye-mask-container", is_img2img)):
                        result = self.render_inner(is_img2img)
        else:
            with gr.Blocks():
                with gr.Group(elem_id=self.get_id("eye-mask-container", is_img2img)):
                    result = self.render_inner(is_img2img)
        return result

    def render_inner(self, is_img2img):

        def get_id(id):
            return self.get_id(id, is_img2img)

        result = []

        if self.is_embedded:
            with gr.Row():
                with gr.Column(scale=2):
                    em_enabled = gr.Checkbox(elem_id=get_id("enabled"), label="Enable", value=False, visible=True)
                    result.append(em_enabled)
                with gr.Column(scale=1):
                    em_n_iter = gr.Slider(elem_id=get_id("count"), label="Batch count", minimum=1, maximum=100, step=1, value=1, visible=True)
                    result.append(em_n_iter)

        with gr.Row():
            em_mask_prompt = gr.Textbox(
                elem_id=get_id("prompt"),
                show_label=False,
                lines=1,
                placeholder="Mask prompt",
                visible=True
            )
        with gr.Row():
            em_mask_negative_prompt = gr.Textbox(
                elem_id=get_id("negative-prompt"),
                show_label=False,
                lines=1,
                placeholder="Negative mask prompt",
                visible=True
            )
        with gr.Row():
            em_mask_type = gr.Radio(
                elem_id=get_id("mask-type"),
                label="Mask type",
                choices=self.eye_mask_core.MASK_TYPES,
                value=self.eye_mask_core.MASK_TYPES[0],
                type="index"
            )
            em_info = ToolButton(value="\u2139\uFE0F", elem_id=get_id("eye-info-button"), full_width=False)
        with gr.Row(elem_id=get_id("mask-type-row")):
            with gr.Accordion("Mask Preview", open=False):
                with gr.Row():
                    em_mp_input_image = gr.Image(source="upload", mirror_webcam=False, type="pil")
                    em_mp_generated_image = gr.Image(label="Mask result", visible=True)
                with gr.Row():
                    def on_generate_mask_click(input_image, em_mask_type):
                        if input_image is None:
                            return
                        mask, mask_success = self.eye_mask_core.generate_mask(input_image, em_mask_type)
                        return gr.update(value=mask, visible=mask_success, interactive=False)
                    em_mp_generate_button = gr.Button(value="Generate mask")
                    em_mp_generate_button.click(fn=on_generate_mask_click, inputs=[em_mp_input_image, em_mask_type], outputs=[em_mp_generated_image])
        with gr.Row():
            em_mask_padding = gr.Slider(elem_id=get_id("mask-padding"), label="Mask padding (dlib only)", minimum=0, maximum=100, step=1, value=20, visible=True)
            em_mask_steps = gr.Slider(elem_id=get_id("mask-steps"), minimum=1, maximum=150, step=1, label="Sampling steps", value=20)
        with gr.Row():
            em_mask_blur = gr.Slider(elem_id=get_id("mask-blur"), label="Mask blur", minimum=0, maximum=64, step=1, value=4, visible=True)
            em_denoising_strength = gr.Slider(elem_id=get_id("denoising-strength"), label="Denoising strength (Inpaint)", minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=True)
        with gr.Row():
            em_inpaint_full_res = gr.Checkbox(label="Inpaint at full resolution", value=True, visible=False)
            em_inpaint_full_res_padding = gr.Slider(elem_id=get_id("full-res-padding"), label="Inpaint at full resolution padding, pixels", minimum=0, maximum=256, step=4, value=88, visible=True)
            em_cfg_scale = gr.Slider(elem_id=get_id("cfg"), minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
        with gr.Row():
            with gr.Column(scale=4):
                em_width = gr.Slider(elem_id=get_id("width"), minimum=64, maximum=2048, step=8, label="Width", value=512)
                em_height = gr.Slider(elem_id=get_id("height"), minimum=64, maximum=2048, step=8, label="Height", value=512)
        with gr.Row():
            em_include_mask = gr.Checkbox(elem_id=get_id("include-mask"), label="Include mask", value=True, visible=True)

            if not self.is_embedded:
                em_redraw_original = gr.Checkbox(elem_id=get_id("redraw-original"), label="Redraw original", value=True, visible=(not is_img2img))
                result.append(em_redraw_original)

            em_mask_padding_in_px = gr.Checkbox(elem_id=get_id("padding-in-px"), label="Padding in px", value=True, visible=True)
            em_use_other_model = gr.Checkbox(elem_id=get_id("use-other-model"), label="Use other model", value=False, visible=True)
        with gr.Row():
            em_model = gr.Dropdown(
                elem_id=get_id('mask-model'),
                label="Mask model",
                choices=["None"] + list(modules.sd_models.checkpoints_list.keys()),
                value="None",
                visible=False,
                type="value",
            )
            em_use_other_model.change(
                lambda visible: {"visible": visible, "__type__": "update"},
                inputs=[em_use_other_model],
                outputs=[em_model]
            )
        with gr.Row():
            reload_info = gr.HTML()
        with gr.Row():
            reload_button = gr.Button(
                elem_id=get_id("reload-extension"),
                value="Reload Extension",
                full_width=True,
                visible=utils.get_opt("em_dev_mode")
            )
            reload_button.click(
                fn=self.restart,
                _js="() => toastr.success('Extension reloaded')",
                outputs=[reload_info],
                show_progress=False
            )

        return result + [
            em_mask_type,
            em_mask_prompt,
            em_mask_negative_prompt,
            em_mask_padding,
            em_mask_padding_in_px,
            em_mask_steps,
            em_include_mask,
            em_mask_blur,
            em_denoising_strength,
            em_cfg_scale,
            em_width,
            em_height,
            em_inpaint_full_res,
            em_inpaint_full_res_padding,
            em_use_other_model,
            em_model
        ]
