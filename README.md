<p float="left">
    <img alt="" src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
    <img alt="" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img alt="" src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

# stable-diffusion-webui-eyemask

This extension is for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### Capabilities

* Create mask for eyes / face / body and redraw by multiple params
* Wildcards for mask prompt and original prompt
* Using of different model just for masked area redraw
* Mask padding in px or percents
* Mask previews
* Separate embedded version of the script (enabled from settings)
* Batch mode support
* Custom API routes (including serving static files)

Put all wildcards files in */wildcards* dir.

### Install

Use *Install from URL* option with this repo url.

### Requirements

All requirements will be installed on first use, except for:
 - [cmake](https://cmake.org/download/) - used only for dlib masks

### Mask types

1. Eyes dlib
2. Face dlib
3. Face depthmask
4. Body depthmask
5. Face mmdet
6. Body mmdet

### Examples

<img width="1024" src="https://raw.githubusercontent.com/ilian6806/stable-diffusion-webui-eyemask/master/static/images/mask-types.jpg" alt="">

### Contributing

Feel free to submit PRs to develop!