
function log(m) {
    console.log(m);
}

String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) {
        return typeof args[number] != 'undefined' ? args[number] : match;
    });
};

document.addEventListener('DOMContentLoaded', function() {

    $.fn.bindClick = function (func, args) {
        if (args) {
            return this.off('click').on('click', function () {
                func.apply(this, args);
            });
        } else {
            return this.off('click').on('click', func);
        }
    };

    $.fn.triggerEvent = function (event) {
        if (! this.length) {
            return this;
        }
        let el = this[0];
        event.split(' ').forEach(function (evt) {
            el.dispatchEvent(new Event(evt.trim()));
        });
        return this;
    };
});