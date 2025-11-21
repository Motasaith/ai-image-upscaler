import cv2
import os
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

class ImageRestorer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"AI Restorer initialized on: {self.device}")
        
        # 1. Setup Real-ESRGAN (Background/General Enhancer)
        # We use the x4plus model which is best for general restoration
        model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download it.")

        self.model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=self.model_esrgan,
            tile=400, # Tile size to save VRAM. Decrease to 200 if you crash.
            tile_pad=10,
            pre_pad=0,
            half=True if self.device.type == 'cuda' else False,
            device=self.device
        )

        # 2. Setup GFPGAN (Face Restorer)
        # This is crucial for eyes/mouths that standard upscalers destroy
        face_model_path = os.path.join('weights', 'GFPGANv1.3.pth')
        if not os.path.exists(face_model_path):
             raise FileNotFoundError(f"Model not found at {face_model_path}. Please download it.")

        self.face_enhancer = GFPGANer(
            model_path=face_model_path,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler
        )

    def process_image(self, img_array, face_enhance=True):
        """
        img_array: numpy array (cv2 format)
        face_enhance: bool to enable specific face correction
        """
        try:
            if face_enhance:
                # GFPGANer automatically calls the background upsampler (RealESRGAN) 
                # internally if we provided it in __init__.
                # It then detects faces, restores them, and pastes them back.
                _, _, output = self.face_enhancer.enhance(
                    img_array, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
            else:
                # Just use RealESRGAN without face logic
                output, _ = self.upsampler.enhance(img_array, outscale=4)
                
            return output
        except RuntimeError as e:
            print(f"Error during inference: {e}")
            return None