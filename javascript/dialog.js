
var dialog = (function () {

    var DIALOG_FADE_IN_SPEED = 250;
    var DIALOG_FADE_OUT_SPEED = 150;

    var $mainContainer = null;

    var onClose = null;

    function getContainer() {
        if (! $mainContainer) {
            $mainContainer = $('#dialogs-container');
        }
        return $mainContainer;
    }

    var mainLayout = (
        '<div id="{0}-modal" class="modal" tabindex="-1">' +
            '<div class="modal-header">' +
                '<h5 class="modal-title"></h5>' +
            '</div>' +
            '<div id="{0}-modal-content" class="modal-content"></div>' +
            '<div class="modal-footer"></div>' +
        '</div>'
    );

    function getContent(opt) {
        return mainLayout.format(opt.id);
    }

    function getButton(btnOpt) {
        var template = '<button {0} type="button" class="button {1}">{2}</button>';
        var id = btnOpt.id ? ' id="' + btnOpt.id + '-modal-btn" ' : '';
        var className = btnOpt.className ? btnOpt.className : '';
        var text = btnOpt.text;
        return template.format(id, className, text);
    }

    function close() {
        getContainer()
            .removeClass('open')
            .find('div.modal').fadeOut(DIALOG_FADE_OUT_SPEED);

        if (onClose && onClose.call) {
            onClose();
        }
    }

    function onFooterBtnClick(data) {
        if (data.action && data.action.call) {
            data.action.call(this, data.params || []);
        }
        if (! data.dontClose) {
            close();
        }
    }

    function show(opt) {

        onClose = null;
        close();

        if ((typeof opt).toLowerCase() === 'string') {
            opt = {
                content: opt
            };
        }

        if (! opt.id) {
            opt.id = +new Date;
        }

        onClose = (opt.onClose && opt.onClose.call) ? opt.onClose : null;

        var dialogId = '#{0}-modal'.format(opt.id);

        if (! getContainer().find(dialogId).length) {
            getContainer().append(getContent(opt));
        }

        $dialog = $(dialogId);

        if (opt.big || opt.imageSrc) {
            var windowHeight = $(window).height();
            var minHeight = parseInt(windowHeight * 0.8) - 100;
            var top = parseInt(windowHeight * 0.08);
            $dialog.css('top', top + 'px');
            if (! opt.dontStretch) {
                $dialog.find('.modal-content').css('min-height', minHeight);
            }
            $dialog.addClass('modal-big');

            if (opt.imageSrc) {
                $dialog.find('.modal-content').css({
                    'background-image': 'url(' + opt.imageSrc + ')',
                    'background-size': 'contain',
                    'background-repeat' : 'no-repeat',
                    'background-position': 'center center'
                });
            }
        }

        if (opt.maxWidth) {
            $dialog.css('max-width', opt.maxWidth);
        }

        if (opt.width) {
            $dialog.css('width', opt.width);
        }

        if (opt.top) {
            $dialog.css('top', opt.top);
        }

        $dialog
            .find('.modal-title').html(opt.title).end()
            .find('.modal-content').html(opt.content || opt.text || '').end()
            .bindClick(function (e) {
                e.stopPropagation();
            })
            .fadeIn(DIALOG_FADE_IN_SPEED);

        getContainer().addClass('open');

        if (! opt.title) {
            $dialog
                .find('.modal-header').hide().end()
                .find('.modal-content').css('padding', '30px 20px')
        }

        if (opt.removeWrapper) {
            $dialog.find('.modal-content *').first().unwrap();
        }
        $footer = $dialog.find('.modal-footer').first().empty();

        if (! opt.disableOverlay) {
            getContainer().bindClick(close);
        } else {
            getContainer().off('click');
        }

        if (! opt.buttons || ! opt.buttons.length) {
            opt.buttons = [{
                id: 'modal-close-button',
                text: 'Ok'
            }];
        }

        for (var i = 0, len = opt.buttons.length; i < len; i++) {
            if (! opt.buttons[i].id) {
                opt.buttons[i].id = Math.floor(Math.random() * 1000000);
            }
            $footer
                .append(getButton(opt.buttons[i]))
                .find('#' + opt.buttons[i].id + '-modal-btn')
                .bindClick(onFooterBtnClick, [opt.buttons[i]]);
        }

        return $dialog;
    }

    function showConfirm(text, action) {
        this.show({
            id: 'main-confirmation',
            title: 'Confirmation',
            content: text,
            buttons: [
                {
                    text: 'Yes',
                    className: 'danger',
                    action: action
                }, {
                    text: 'No'
                }
            ]
        });
    }

    function showImage(src, title, buttons, maxWidth) {
        this.show({
            id: 'image-dialog',
            imageSrc: '/sdapi/v1/eyemask/v1/static/images/' + src,
            big: true,
            maxWidth: maxWidth || null,
            title: title || null,
            buttons: buttons || null
        });
    }

    function showError(message) {
        this.show({
            id: 'error-dialog',
            title: 'Error',
            content: message,
        });
    }

    function hide() {
        close();
    }

    return {
        show: show,
        hide: hide,
        confirm: showConfirm,
        image: showImage,
        error: showError,
    };
})();