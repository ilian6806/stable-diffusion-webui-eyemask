import gc
import re

import modules.shared as shared
from modules import devices, images
from modules.processing import fix_seed, process_images, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from . import mask_generator, utils, widlcards
from .state import SharedSettingsContext


class EyeMasksCore():

    # Just comment mask type to disable it
    MASK_TYPES = [
        'Eyes dlib',
        'Face dlib',
        'Face depth',
        'Body depth',
        # 'Face mmdet',
        # 'Body mmdet'
    ]

    MASK_TYPE_EYES_DLIB = utils.index(MASK_TYPES, 'Eyes dlib')
    MASK_TYPE_FACE_DLIB = utils.index(MASK_TYPES, 'Face dlib')
    MASK_TYPE_FACE_DEPTH = utils.index(MASK_TYPES, 'Face depth')
    MASK_TYPE_BODY_DEPTH = utils.index(MASK_TYPES, 'Body depth')
    MASK_TYPE_FACE_MMDET = utils.index(MASK_TYPES, 'Face mmdet')
    MASK_TYPE_BODY_MMDET = utils.index(MASK_TYPES, 'Body mmdet')

    # Replaced in original image generation info with regex on each iteration
    EM_DYNAMIC_PARAMS = [
        'em_mask_prompt_final'
    ]

    def execute(self, p,
        em_redraw_original,
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
    ):
        em_params = {
            'em_mask_prompt': em_mask_prompt,
            'em_mask_negative_prompt': em_mask_negative_prompt,
            'em_mask_type': em_mask_type,
            'em_mask_padding': em_mask_padding,
            'em_mask_steps': em_mask_steps,
            'em_mask_blur': em_mask_blur,
            'em_denoising_strength': em_denoising_strength,
            'em_cfg_scale': em_cfg_scale,
            'em_width': em_width,
            'em_height': em_height,
            'em_inpaint_full_res': em_inpaint_full_res,
            'em_inpaint_full_res_padding': em_inpaint_full_res_padding
        }

        fix_seed(p)
        seed = p.seed
        iterations = p.n_iter
        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        initial_info = None
        orig_image_info = None
        new_txt2img_info = None
        new_img2img_info = None

        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        is_img2img = not is_txt2img

        wildcards_generator_original = widlcards.WildcardsGenerator()
        wildcards_generator_mask = widlcards.WildcardsGenerator()

        if (is_img2img):
            orig_image = p.init_images[0]

            if orig_image.info is not None and 'parameters' in orig_image.info:
                orig_image_info = orig_image.info['parameters']
            init_orig_prompt = p.prompt or ''
        else:
            p_txt = p
            p = StableDiffusionProcessingImg2Img(
                init_images = None,
                resize_mode = 0,
                denoising_strength = em_denoising_strength,
                mask = None,
                mask_blur= em_mask_blur,
                inpainting_fill = 1,
                inpaint_full_res = em_inpaint_full_res,
                inpaint_full_res_padding= em_inpaint_full_res_padding,
                inpainting_mask_invert= 0,
                sd_model=p_txt.sd_model,
                outpath_samples=p_txt.outpath_samples,
                outpath_grids=p_txt.outpath_grids,
                prompt=p_txt.prompt,
                negative_prompt=p_txt.negative_prompt,
                styles=p_txt.styles,
                seed=p_txt.seed,
                subseed=p_txt.subseed,
                subseed_strength=p_txt.subseed_strength,
                seed_resize_from_h=p_txt.seed_resize_from_h,
                seed_resize_from_w=p_txt.seed_resize_from_w,
                sampler_name=p_txt.sampler_name,
                n_iter=p_txt.n_iter,
                steps=p_txt.steps,
                cfg_scale=p_txt.cfg_scale,
                width=p_txt.width,
                height=p_txt.height,
                tiling=p_txt.tiling,
            )
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            init_orig_prompt = p_txt.prompt or ''

        output_images = []
        init_image = None
        mask = None
        mask_success = False
        shared.state.job_count = 0

        changing_model = em_use_other_model and em_model != 'None'

        if changing_model:
            em_params['em_mask_model'] = em_model

        with SharedSettingsContext(changing_model) as context:

            for n in range(iterations):

                devices.torch_gc()
                gc.collect()

                start_seed = seed + n
                new_image_generated = False

                mask_prompt = em_mask_prompt
                if em_mask_prompt is not None and len(em_mask_prompt.strip()) > 0:
                    mask_prompt = wildcards_generator_mask.build_prompt(em_mask_prompt)

                em_params['em_mask_prompt_final'] = mask_prompt

                if is_txt2img:
                    if init_image is None or em_redraw_original:
                        p_txt.seed = start_seed
                        init_image, new_txt2img_info, new_image_generated = self.create_new_image(
                            p_txt, em_params, init_orig_prompt, changing_model, context, wildcards_generator_original
                        )
                else:
                    if init_image is None:
                        init_image, new_img2img_info, new_image_generated = self.create_new_image(
                            p, em_params, init_orig_prompt, changing_model, context, wildcards_generator_original
                        )

                p.seed = start_seed
                p.init_images = [init_image]
                p.prompt = mask_prompt
                p.negative_prompt = em_mask_negative_prompt

                if new_image_generated:
                    mask, mask_success = self.get_mask(
                        em_mask_type, em_mask_padding, em_mask_padding_in_px, init_image,
                        p, start_seed, mask_prompt, initial_info
                    )

                if mask_success:
                    p.image_mask = mask

                p.steps = em_mask_steps
                p.denoising_strength = em_denoising_strength
                p.mask_blur = em_mask_blur
                p.cfg_scale = em_cfg_scale
                p.width = em_width
                p.height = em_height
                p.inpaint_full_res = em_inpaint_full_res
                p.inpaint_full_res_padding= em_inpaint_full_res_padding
                p.inpainting_mask_invert = 0

                print(f"Processing {n + 1} / {iterations}.")

                if changing_model:
                    context.apply_checkpoint(em_model)

                shared.state.job_count += 1
                processed = process_images(p)
                save_prompt = p.prompt

                if is_txt2img:
                    initial_info = new_txt2img_info
                    save_prompt = p_txt.prompt
                elif not is_txt2img:
                    initial_info = new_img2img_info
                    try:
                        save_prompt = orig_image_info.split('\n')[0]
                    except Exception as e:
                        print(e)
                        save_prompt = orig_image_info

                updated_info = self.update_info(initial_info, em_params)

                output_images.append(processed.images[0])
                try:
                    p.all_seeds.append(start_seed)
                    p.all_prompts.append(save_prompt)
                    p.infotexts.append(updated_info)
                except Exception as e:
                    pass

                if em_include_mask and (n == iterations - 1 or (is_txt2img and em_redraw_original)):
                    output_images.append(mask)
                    try:
                        p.all_seeds.append(start_seed)
                        p.all_prompts.append(mask_prompt)
                        p.infotexts.append(updated_info)
                    except Exception as e:
                        pass

                shared.state.current_image = processed.images[0]

                if shared.opts.samples_save:
                    images.save_image(
                        processed.images[0],
                        p.outpath_samples,
                        "",
                        start_seed,
                        save_prompt,
                        shared.opts.samples_format,
                        info=updated_info,
                        p=p
                    )

        devices.torch_gc()
        gc.collect()

        return Processed(p, output_images, seed, initial_info)

    def generate_mask(self, init_image, em_mask_type, em_mask_padding=20, em_mask_padding_in_px=False):
        if em_mask_type == self.MASK_TYPE_FACE_DLIB:
            return mask_generator.get_face_mask_dlib(init_image, em_mask_padding, em_mask_padding_in_px)
        elif em_mask_type == self.MASK_TYPE_FACE_DEPTH:
            return mask_generator.get_face_mask_depth(init_image)
        elif em_mask_type == self.MASK_TYPE_BODY_DEPTH:
            return mask_generator.get_body_mask_depth(init_image)
        elif em_mask_type == self.MASK_TYPE_FACE_MMDET:
            return mask_generator.get_face_mask_mmdet(init_image)
        elif em_mask_type == self.MASK_TYPE_BODY_MMDET:
            return mask_generator.get_body_mask_mmdet(init_image)
        else:
            return mask_generator.get_eyes_mask_dlib(init_image, em_mask_padding, em_mask_padding_in_px)

    def get_mask(self,
        em_mask_type, em_mask_padding, em_mask_padding_in_px,
        init_image, p, start_seed, mask_prompt, initial_info
    ):
        mask, mask_success = self.generate_mask(init_image, em_mask_type, em_mask_padding, em_mask_padding_in_px)

        if shared.opts.em_save_masks:
            images.save_image(
                mask,
                shared.opts.em_outdir_masks,
                "",
                start_seed,
                mask_prompt,
                shared.opts.samples_format,
                info=initial_info,
                p=p
            )

        return mask, mask_success

    def update_info(self, info, em_params):
        reg_ex = ':\s[0-9a-zA-Z\-\.\s]+'
        for param in self.EM_DYNAMIC_PARAMS:
            if param in em_params:
                info = re.sub(param + reg_ex, '%s: %s' % (param, em_params[param]), info)
        return info

    ##############################
    ##### Creating new image #####
    ##############################

    def create_new_image(self, p, em_params, init_orig_prompt, changing_model, context, wildcards_generator):
        em_params = utils.removeEmptyStringValues(em_params)
        if changing_model:
            context.restore_original_checkpoint()
        self.build_original_prompt(p, init_orig_prompt, em_params, wildcards_generator)
        return self.generate_initial_image_with_extra_params(p, em_params)

    def build_original_prompt(self, p, init_orig_prompt, em_params, wildcards_generator):
        if not shared.opts.em_wildcards_in_original:
            return
        new_prompt = wildcards_generator.build_prompt(init_orig_prompt)
        if new_prompt != init_orig_prompt:
            em_params['em_prompt'] = init_orig_prompt
            em_params['em_prompt_final'] = new_prompt
            p.prompt = new_prompt

    def generate_initial_image_with_extra_params(self, p, extra_params):
        p.extra_generation_params = p.extra_generation_params or {}
        p.extra_generation_params.update(extra_params)
        shared.state.job_count += 1
        processed = process_images(p)
        return processed.images[0], processed.info, True
