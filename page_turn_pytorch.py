import torch
import numpy as np
from PIL import Image
import math

class PageTurnEffect:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.PI = 3.14159265359
        self.FOLD_RADIUS = 0.3
        self.OPACITY = 1
        self.EPSILON = 1e-5
        self.device = device

    def process_image(self, input_image, mouse_pos):
        # 将输入图像转换为 RGBA 格式
        input_image = input_image.convert('RGBA')
        img_tensor = torch.from_numpy(np.array(input_image)).float().to(self.device) / 255.0
        height, width, _ = img_tensor.shape
        resolution = torch.tensor([width, height], device=self.device, dtype=torch.float32)
        aspect_ratio = width / height
        inv_aspect = torch.tensor([1.0 / aspect_ratio, 1.0], device=self.device, dtype=torch.float32)
        
        output = img_tensor.clone()
        
        fold_position = torch.tensor(mouse_pos, device=self.device, dtype=torch.float32)
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
        
        mask_outside = dist > self.FOLD_RADIUS
        mask_curl = (dist > self.EPSILON) & ~mask_outside
        
        theta = torch.arcsin(torch.clamp(dist[mask_curl] / self.FOLD_RADIUS, -1, 1))
        p1 = line_point[mask_curl] - dir_vector * theta.unsqueeze(-1) * self.FOLD_RADIUS
        p2 = line_point[mask_curl] - dir_vector * (self.PI - theta).unsqueeze(-1) * self.FOLD_RADIUS
        
        p1_coords = (p1 * inv_aspect * resolution).long()
        p2_coords = (p2 * inv_aspect * resolution).long()
        
        output[mask_outside] = 0
        
        valid_coords = (p1_coords[:, 0] >= 0) & (p1_coords[:, 0] < width) & \
                      (p1_coords[:, 1] >= 0) & (p1_coords[:, 1] < height) & \
                      (p2_coords[:, 0] >= 0) & (p2_coords[:, 0] < width) & \
                      (p2_coords[:, 1] >= 0) & (p2_coords[:, 1] < height)
        
        p1_valid = p1_coords[valid_coords]
        p2_valid = p2_coords[valid_coords]
        curl_positions = torch.nonzero(mask_curl)[valid_coords]
        
        front_color = img_tensor[p1_valid[:, 1], p1_valid[:, 0]]
        back_color = img_tensor[p2_valid[:, 1], p2_valid[:, 0]]
        
        # 提取 Alpha 通道
        front_alpha = front_color[..., 3]  # 正面颜色的 Alpha 通道
        back_alpha = back_color[..., 3]    # 背面颜色的 Alpha 通道

        # 判断背面是否透明
        is_back_transparent = back_alpha < self.EPSILON  # 如果 Alpha 接近 0，则认为透明
        
        # 初始化混合颜色
        mixed_color = torch.zeros_like(front_color)
        
        # 如果背面透明，直接使用正面颜色
        mixed_color[is_back_transparent] = front_color[is_back_transparent]
        
        # 如果背面不透明，按照原来的逻辑混合颜色
        not_transparent = ~is_back_transparent
        if torch.any(not_transparent):  # 确保不透明区域存在
            # 提取不透明区域的颜色
            front_color_not_transparent = front_color[not_transparent]
            back_color_not_transparent = back_color[not_transparent]
            light_not_transparent = ((self.FOLD_RADIUS - dist[mask_curl][valid_coords][not_transparent]) / self.FOLD_RADIUS) * 0.6
        
            # 应用光照效果
            back_color_not_transparent = back_color_not_transparent + light_not_transparent.unsqueeze(-1)
            back_color_not_transparent = torch.clamp(back_color_not_transparent, 0, 1)  # 确保颜色值在[0,1]之间
        
            # 混合颜色
            mixed_color_not_transparent = torch.clamp(
                front_color_not_transparent * (1 - self.OPACITY) + back_color_not_transparent * self.OPACITY,
                0, 1
            ).float()
        
            # 将混合颜色赋值给不透明区域
            mixed_color[not_transparent] = mixed_color_not_transparent

        # 将混合颜色赋值给输出图像
        output[curl_positions[:, 0], curl_positions[:, 1]] = mixed_color.to(output.dtype)
        
        output_np = (output.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(output_np)

class PageTurn:
    def __init__(self, input_image, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.effect = PageTurnEffect(device=device)
        self.input_image = input_image
        self.width, self.height = self.input_image.size
    
    @staticmethod
    def ease_in_out_cubic(x):
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def calculate_fold_position(self, progress):
        progress = max(0.0, min(1.0, progress))
        start_pos = np.array([self.width, 0])
        end_pos = np.array([0, self.height])
        eased_progress = self.ease_in_out_cubic(progress)
        current_pos = start_pos + (end_pos - start_pos) * eased_progress
        return current_pos

    def generate_frame(self, progress):
        fold_pos = self.calculate_fold_position(progress)
        return self.effect.process_image(self.input_image, fold_pos)

    def generate_animation_frames(self, num_frames=60):
        frames = []
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            frame = self.generate_frame(progress)
            frames.append(frame)
        return frames

def main():
    input_image = Image.open('input2.png').convert('RGBA')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    page_turn = PageTurn(input_image, device=device)
    progress = 0.8
    result_image = page_turn.generate_frame(progress)
    result_image.save('output_frame.png')
    frames = page_turn.generate_animation_frames(num_frames=30)
    frames[0].save('animation.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    return result_image

result = main()
result  # 在Jupyter notebook中显示结果图片
