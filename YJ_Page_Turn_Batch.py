import torch
import numpy as np
from PIL import Image
import math

class PageTurnNode:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PI = 3.14159265359
        self.EPSILON = 1e-5

    @staticmethod
    def tensor2pil(image: torch.Tensor) -> Image.Image:
        """
        Convert a (H, W, C) tensor to a PIL image.
        """
        image_np = np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a (H, W, C) tensor.
        """
        img_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "process": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fold_radius": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.01}),
                "light_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_page_turn"
    CATEGORY = "YJ"

    @staticmethod
    def ease_in_out_cubic(x):
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def calculate_fold_position(self, width, height, progress):
        """
        Calculate the fold position based on the progress (0-100).
        """
        progress = max(0.0, min(100.0, progress)) / 100.0  # Normalize to 0-1
        start_pos = np.array([width, 0])
        end_pos = np.array([0, height])
        eased_progress = self.ease_in_out_cubic(progress)
        current_pos = start_pos + (end_pos - start_pos) * eased_progress
        return current_pos

    def process_image(self, input_image, fold_pos, opacity, fold_radius, light_intensity):
        """
        Process a single image to apply the page turn effect.
        """
        # Convert (H, W, C) tensor to (C, H, W) for processing
        img_tensor = input_image.permute(2, 0, 1).to(self.device)
        C, height, width = img_tensor.shape
        resolution = torch.tensor([width, height], device=self.device, dtype=torch.float32)
        aspect_ratio = width / height
        inv_aspect = torch.tensor([1.0 / aspect_ratio, 1.0], device=self.device, dtype=torch.float32)
        
        output = img_tensor.clone()
        
        fold_position = torch.tensor(fold_pos, device=self.device, dtype=torch.float32)
        corner_position = torch.tensor([width, 0.0], device=self.device, dtype=torch.float32)
        
        dir_vector = fold_position - corner_position
        dir_vector_norm = torch.norm(dir_vector) + self.EPSILON
        dir_vector = dir_vector / dir_vector_norm
        
        y, x = torch.meshgrid(torch.arange(height, device=self.device, dtype=torch.float32), 
                              torch.arange(width, device=self.device, dtype=torch.float32), indexing='ij')
        uv = torch.stack([x, y], dim=-1) / resolution
        uv[..., 0] *= aspect_ratio
        
        normalized_fold_pos = fold_position * torch.tensor([aspect_ratio, 1.0], device=self.device, dtype=torch.float32) / resolution
        
        origin = normalized_fold_pos - dir_vector * (
            (normalized_fold_pos[0] - aspect_ratio * torch.where(dir_vector[0] > -self.EPSILON, 1.0, 0.0)) / 
            (dir_vector[0] + self.EPSILON)
        )
        
        diff = origin.unsqueeze(0).unsqueeze(0) - uv
        proj = (diff * dir_vector).sum(dim=-1)
        curl_dist = torch.norm(normalized_fold_pos - origin)
        dist = proj - curl_dist
        
        line_point = uv + dist.unsqueeze(-1) * dir_vector
        
        mask_outside = dist > fold_radius
        mask_curl = (dist > self.EPSILON) & ~mask_outside
        
        theta = torch.arcsin(torch.clamp(dist[mask_curl] / fold_radius, -1, 1))
        p1 = line_point[mask_curl] - dir_vector * theta.unsqueeze(-1) * fold_radius
        p2 = line_point[mask_curl] - dir_vector * (self.PI - theta).unsqueeze(-1) * fold_radius
        
        p1_coords = (p1 * inv_aspect * resolution).long()
        p2_coords = (p2 * inv_aspect * resolution).long()
        
        output[:, mask_outside] = 0
        
        valid_coords = (p1_coords[:, 0] >= 0) & (p1_coords[:, 0] < width) & \
                      (p1_coords[:, 1] >= 0) & (p1_coords[:, 1] < height) & \
                      (p2_coords[:, 0] >= 0) & (p2_coords[:, 0] < width) & \
                      (p2_coords[:, 1] >= 0) & (p2_coords[:, 1] < height)
        
        p1_valid = p1_coords[valid_coords]
        p2_valid = p2_coords[valid_coords]
        curl_positions = torch.nonzero(mask_curl)[valid_coords]
        
        front_color = img_tensor[:, p1_valid[:, 1], p1_valid[:, 0]]
        back_color = img_tensor[:, p2_valid[:, 1], p2_valid[:, 0]]
        
        front_alpha = front_color[3] if C == 4 else torch.ones_like(front_color[0])
        back_alpha = back_color[3] if C == 4 else torch.ones_like(back_color[0])
        
        is_back_transparent = back_alpha < self.EPSILON
        
        mixed_color = torch.zeros_like(front_color)
        mixed_color[:, is_back_transparent] = front_color[:, is_back_transparent]
        
        not_transparent = ~is_back_transparent
        if torch.any(not_transparent):
            front_color_not_transparent = front_color[:, not_transparent]
            back_color_not_transparent = back_color[:, not_transparent]
            light_not_transparent = ((fold_radius - dist[mask_curl][valid_coords][not_transparent]) / fold_radius) * light_intensity
            
            # Reshape light_not_transparent to match back_color_not_transparent
            light_not_transparent = light_not_transparent.unsqueeze(0).expand(C, -1)
            
            back_color_not_transparent = back_color_not_transparent + light_not_transparent
            back_color_not_transparent = torch.clamp(back_color_not_transparent, 0, 1)
            
            mixed_color_not_transparent = torch.clamp(
                front_color_not_transparent * (1 - opacity) + back_color_not_transparent * opacity,
                0, 1
            ).float()
            
            mixed_color[:, not_transparent] = mixed_color_not_transparent

        output[:, curl_positions[:, 0], curl_positions[:, 1]] = mixed_color.to(output.dtype)
        
        # Convert back to (H, W, C)
        output = output.permute(1, 2, 0)
        output_np = (output.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(output_np)

    def apply_page_turn(self, image, process, opacity, fold_radius, light_intensity):
        """
        Apply the page turn effect to a batch of images.
        """
        batch_size = image.shape[0]
        processed_images = []
        for i in range(batch_size):
            img_tensor = image[i]  # (H, W, C)
            assert img_tensor.dim() == 3, f"Image tensor should have 3 dimensions, but has {img_tensor.dim()}"
            assert img_tensor.shape[2] in [3, 4], f"Image tensor should have 3 or 4 channels, but has {img_tensor.shape[2]}"
            
            img_pil = self.tensor2pil(img_tensor)
            width, height = img_pil.size
            fold_pos = self.calculate_fold_position(width, height, process)
            processed_img_pil = self.process_image(img_tensor, fold_pos, opacity, fold_radius, light_intensity)
            processed_img_tensor = self.pil2tensor(processed_img_pil)
            processed_images.append(processed_img_tensor)
        
        processed_image_batch = torch.stack(processed_images)
        return (processed_image_batch,)
        
NODE_CLASS_MAPPINGS = {
    "YJPageTurn": PageTurnNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YJPageTurn": "YJ Page Turn Batch"
}
