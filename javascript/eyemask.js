
document.addEventListener('DOMContentLoaded', function() {

    window.BODY_SELECTOR = '.mx-auto.container'

    toastr.options = {
        target: window.BODY_SELECTOR,
        timeOut: 3500
    };

    window.gradioRoot = gradioApp().querySelector('.gradio-container');
    window.$gradioRoot = $(window.gradioRoot);

    onUiLoaded(EyeMaskController.load);

    $gradioRoot.append('<div id="dialogs-container"></div>')
});


const EyeMaskController = (function () {

    let container = null;
    let config = {};

    const LS_PREFIX = 'em-save-';
    const TABS = ['txt2img', 'img2img'];

    const emTitles = {
        '\u21A9\uFE0F': 'Load current image eyemask params',
        'Redraw original': 'Change seed after each batch',
        'Include mask': 'Include mask image in result'
    };

    function getContainer() {
        if (!container) {
            container = gradioApp().getElementById('eye-mask-container');
        }
        return container
    }

    function getApiUrl(path) {
        return '/sdapi/v1/eyemask/v1' + path;
    }

    function loadTitles() {
        gradioApp().querySelectorAll('span, button, select, p').forEach(function(elem) {
            if (elem) {
                let tooltip = emTitles[elem.textContent] || emTitles[elem.value];
                if (tooltip) {
                    elem.title = tooltip;
                }
            }
        });
    }

    function getConfig() {
        $.ajax({
            url: getApiUrl('/config.json'),
            dataType: 'json',
            async: false,
            cache: false,
            success: function(data) {
                config = data;
            }
        });
    }

    function getAllIds(id) {
        let result = [];
        result.push('em-{0}-txt2img'.format(id));
        result.push('em-{0}-img2img'.format(id));
        result.push('em-emb-{0}-txt2img'.format(id));
        result.push('em-emb-{0}-img2img'.format(id));
        return result;
    }

    function loadPlaceHolders() {
        if (config.em_save_prompts) {
            ['txt2img_prompt', 'img2img_prompt'].forEach(handleSavedInput);
        }
        if (config.em_save_neg_prompts) {
            ['txt2img_neg_prompt', 'img2img_neg_prompt'].forEach(handleSavedInput);
        }
        if (config.em_save_em_prompts) {
            getAllIds('prompt').forEach(handleSavedInput);
        }
        if (config.em_save_em_neg_prompts) {
            getAllIds('negative-prompt').forEach(handleSavedInput);
        }
        if (config.em_save_settings) {
            [
                'enabled',
                'count',
                'mask-type',
                'mask-padding',
                'mask-steps',
                'mask-blur',
                'denoising-strength',
                'full-res-padding',
                'cfg',
                'width',
                'height',
                'include-mask',
                'padding-in-px',
                'redraw-original',
                'use-other-model',
                'mask-model'
            ].forEach(function (id) {
                getAllIds(id).forEach(handleSavedInput);
            });
        }
        if (config.em_save_last_script) {
            TABS.forEach(loadLastScript)
        }
    }

    function handleSavedInput(id) {

        let $el = $('#{0} textarea, #{0} select, #{0} input'.format(id));
        let event = 'change input';

        if (! $el.length) {
            return;
        }

        let value = localStorage.getItem(LS_PREFIX + id);

        if (value) {
            switch ($el[0].type) {
                case 'checkbox':
                    $el.prop('checked', value === 'true').triggerEvent(event);
                    break;
                case 'radio':
                    $el.filter(':checked').prop('checked', false);
                    $el.filter('[value="{0}"]'.format(value)).prop('checked', true).triggerEvent(event);
                    break;
                default:
                    $el.val(value).triggerEvent(event);
            }
        }

        $el.on(event,function () {
            let value = this.value;
            if (this.type && this.type === 'checkbox') {
                value = this.checked;
            }
            localStorage.setItem(LS_PREFIX + id, value);
        });

        if (id.indexOf('emb-enabled') > -1 && value === 'true') {
            $('#' + id.replace('em-emb-enabled-', '') + '_script_container .cursor-pointer').triggerEvent('click');
        }

        if (id.indexOf('emb-use-other-model') > -1) {
            setTimeout(function () {
                $el.triggerEvent(event);
            }, 0);
        }
    }

    function loadLastScript(tab) {

        let $select = $('#{0}_script_container #script_list select'.format(tab));
        let value = localStorage.getItem(LS_PREFIX + 'last-script-' + tab);

        $select.on('change', function () {
            localStorage.setItem(LS_PREFIX + 'last-script-' + tab, this.value);
        });

        if (value) {
            setTimeout(function () {
                $select.val(value).triggerEvent('change');
            },0);
        }
    }

    function onFirstLoad() {
        getConfig();
        loadTitles();
        loadPlaceHolders();
    }

    function load() {
        container = getContainer();
        onFirstLoad();
    }

    function showInfo() {
        dialog.image('mask-types.jpg', 'Mask Types', null, '80%');
    }

    return {
        load,
        showInfo
    };
}());
