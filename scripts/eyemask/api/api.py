import os

import gradio as gr
from fastapi import FastAPI, Body, HTTPException, Request, Response
from fastapi.responses import FileResponse

from .. constants import script_static_dir
from .. import script as eye_mask_script
from . utils import encode_pil_to_base64, decode_base64_to_image
from . models import *


class EyeMaskApi():

    def __init__(self):
        self.core = eye_mask_script.EyeMasksCore()

    BASE_PATH = '/sdapi/v1/eyemask'
    VERSION = 1

    def get_path(self, path):
        return f"{self.BASE_PATH}/v{self.VERSION}{path}"

    def add_api_route(self, path: str, endpoint, **kwargs):
        # authenticated requests can be implemented here
        return self.app.add_api_route(self.get_path(path), endpoint, **kwargs)

    def start(self, _: gr.Blocks, app: FastAPI):

        self.app = app

        self.add_api_route('/mask_list', self.mask_list, methods=['GET'])
        self.add_api_route('/generate_mask', self.generate_mask, methods=['POST'], response_model=SingleImageResponse)
        self.add_api_route('/static/{path:path}', self.static, methods=['GET'])
        self.add_api_route('/config.json', self.get_config, methods=['GET'])

    def mask_list(self):
        ''' Get masks list '''
        return { 'mask_list': list(eye_mask_script.EyeMasksCore.MASK_TYPES) }

    def generate_mask(self, req: GenerateMaskRequest):
        ''' Generate mask by type '''
        image = decode_base64_to_image(req.image)
        mask, mask_success = self.core.generate_mask(image, int(req.mask_type))
        return SingleImageResponse(image=encode_pil_to_base64(mask))

    def static(self, path: str):
        ''' Serve static files '''
        static_file = os.path.join(script_static_dir, path)
        if static_file is not None:
            return FileResponse(static_file)
        raise HTTPException(status_code=404, detail='Static file not found')

    def get_config(self):
        return FileResponse('config.json')
