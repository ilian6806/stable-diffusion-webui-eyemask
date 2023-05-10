import modules.shared as shared
from modules import scripts

def on_ui_settings():
    section = ('eyemask', 'Eye Mask')
    options = [
        ('em_show_embedded_version', False, 'Show embedded version'),
        ('em_save_masks', False, 'Save masks'),
        ('em_outdir_masks', 'extensions/eyemask/outputs/masks', 'Output directory for masks'),
        ('em_wildcards_in_original', True, 'Replace wildcards in original prompt'),
        ('em_save_prompts', False, 'Save last prompt'),
        ('em_save_neg_prompts', False, 'Save last negative prompt'),
        ('em_save_em_prompts', False, 'Save last mask prompt'),
        ('em_save_em_neg_prompts', False, 'Save last mask negative prompt'),
        ('em_save_last_script', False, 'Save last script'),
        ('em_save_settings', False, 'Save all settings'),
        ('em_dev_mode', False, 'Dev mode'),
    ]
    for opt in options:
        shared.opts.add_option(opt[0], shared.OptionInfo(opt[1], opt[2], section=section))

scripts.script_callbacks.on_ui_settings(on_ui_settings)
