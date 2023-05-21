import gc

import modules.shared as shared
from modules import devices, images
from modules.processing import fix_seed, process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from . import widlcards
from .state import SharedSettingsContext

from .script import EyeMasksCore


class EyeMasksEmbeddedCore(EyeMasksCore):

    def execute_process(self, *args):

        p, em_enabled = args[:2]

        if not em_enabled:
            return

        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

    def execute_postprocess(self, p, processed,
        em_enabled,
        em_n_iter,
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
        if not em_enabled:
            return

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
        iterations = em_n_iter

        initial_info = None
        orig_image_info = None
        new_img2img_info = None

        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)

        wildcards_generator_original = widlcards.WildcardsGenerator()
        wildcards_generator_mask = widlcards.WildcardsGenerator()

        p_em = StableDiffusionProcessingImg2Img(
            init_images=[processed.images[0]],
            resize_mode=0,
            denoising_strength=em_denoising_strength,
            mask=None,
            mask_blur=em_mask_blur,
            inpainting_fill=1,
            inpaint_full_res=em_inpaint_full_res,
            inpaint_full_res_padding=em_inpaint_full_res_padding,
            inpainting_mask_invert=0,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            styles=p.styles,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            n_iter=p.n_iter,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            tiling=p.tiling,
        )
        p_em.do_not_save_grid = True
        p_em.do_not_save_samples = True

        init_orig_prompt = p.prompt or ''
        initial_info = processed.info

        shared.state.job_count = 0

        changing_model = em_use_other_model and em_model != 'None'

        if changing_model:
            em_params['em_mask_model'] = em_model

        with SharedSettingsContext(changing_model) as context:

            for i in range(len(processed.images)):

                orig_image = processed.images[i]
                init_image = None

                if orig_image.info is not None and 'parameters' in orig_image.info:
                    orig_image_info = orig_image.info['parameters']

                shared.state.job_count += 1

                mask, mask_success = self.get_mask(
                    em_mask_type, em_mask_padding, em_mask_padding_in_px, orig_image,
                    p_em, seed, em_mask_prompt, initial_info
                )

                if mask_success:
                    p_em.image_mask = mask

                for n in range(iterations):

                    devices.torch_gc()
                    gc.collect()

                    start_seed = seed + n

                    mask_prompt = em_mask_prompt
                    if em_mask_prompt is not None and len(em_mask_prompt.strip()) > 0:
                        mask_prompt = wildcards_generator_mask.build_prompt(em_mask_prompt)

                    em_params['em_mask_prompt_final'] = mask_prompt

                    if init_image is None:
                        init_image, new_img2img_info, new_image_generated = self.create_new_image(
                            p_em, em_params, init_orig_prompt, changing_model, context, wildcards_generator_original
                        )

                    p_em.seed = start_seed
                    p_em.init_images = [orig_image]
                    p_em.prompt = mask_prompt
                    p_em.negative_prompt = em_mask_negative_prompt
                    p_em.steps = em_mask_steps
                    p_em.denoising_strength = em_denoising_strength
                    p_em.mask_blur = em_mask_blur
                    p_em.cfg_scale = em_cfg_scale
                    p_em.width = em_width
                    p_em.height = em_height
                    p_em.inpaint_full_res = em_inpaint_full_res
                    p_em.inpaint_full_res_padding= em_inpaint_full_res_padding
                    p_em.inpainting_mask_invert = 0

                    print(f"Processing {n + 1} / {iterations}.")

                    if changing_model:
                        context.apply_checkpoint(em_model)

                    shared.state.job_count += 1
                    processed_em = process_images(p_em)

                    lines = new_img2img_info.splitlines(keepends=True)
                    lines[0] = p.prompt + '\n'
                    new_img2img_info = ''.join(lines)

                    if is_txt2img:
                        initial_info = new_img2img_info
                        save_prompt = p.prompt
                    elif not is_txt2img:
                        initial_info = new_img2img_info
                        try:
                            save_prompt = orig_image_info.split('\n')[0]
                        except Exception as e:
                            print(e)
                            save_prompt = orig_image_info

                    updated_info = self.update_info(initial_info, em_params)

                    processed.images.append(processed_em.images[0])
                    processed.all_seeds.append(start_seed)
                    processed.all_prompts.append(save_prompt)
                    processed.infotexts.append(updated_info)

                    if em_include_mask and (n == iterations - 1):
                        processed.images.append(mask)
                        processed.all_seeds.append(start_seed)
                        processed.all_prompts.append(mask_prompt)
                        processed.infotexts.append(updated_info)

                    shared.state.current_image = processed_em.images[0]

                    if shared.opts.samples_save:
                        images.save_image(
                            processed_em.images[0],
                            p_em.outpath_samples,
                            "",
                            start_seed,
                            save_prompt,
                            shared.opts.samples_format,
                            info=updated_info,
                            p=p_em
                        )

        devices.torch_gc()
        gc.collect()
