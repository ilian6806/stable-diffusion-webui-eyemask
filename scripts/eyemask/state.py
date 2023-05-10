import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae


class SharedSettingsContext(object):

    def __init__(self, changing_model):
        self.changing_model = changing_model

    def __enter__(self):
        if self.changing_model:
            self.CLIP_stop_at_last_layers = shared.opts.CLIP_stop_at_last_layers
            self.sd_model_checkpoint = shared.opts.sd_model_checkpoint
            self.vae = shared.opts.sd_vae
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.changing_model:
            shared.opts.data["sd_vae"] = self.vae
            shared.opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers
            self.apply_checkpoint(self.sd_model_checkpoint)
            modules.sd_vae.reload_vae_weights()

    def apply_checkpoint(self, x):
        info = modules.sd_models.get_closet_checkpoint_match(x)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")
        modules.sd_models.reload_model_weights(shared.sd_model, info)

    def restore_original_checkpoint(self):
        self.apply_checkpoint(self.sd_model_checkpoint)

    def find_vae(self, name: str):
        if name.lower() in ['auto', 'automatic']:
            return modules.sd_vae.unspecified
        if name.lower() == 'none':
            return None
        else:
            choices = [x for x in sorted(modules.sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
            if len(choices) == 0:
                print(f"No VAE found for {name}; using automatic")
                return modules.sd_vae.unspecified
            else:
                return modules.sd_vae.vae_dict[choices[0]]
