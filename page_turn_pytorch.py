import torch
import numpy as np
from PIL import Image
import math
import io

class PageTurnEffect:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 常量定义
        self.PI = 3.14159265359
        self.FOLD_RADIUS = 0.2
        self.OPACITY = 1
        self.EPSILON = 1e-5
        self.device = device

    def if_less_than(self, val, compare_val):
        return torch.maximum(torch.sign(compare_val - val), torch.tensor(0.0, device=self.device, dtype=torch.float32))

    def if_greater_than(self, val, compare_val):
        return torch.maximum(torch.sign(val - compare_val), torch.tensor(0.0, device=self.device, dtype=torch.float32))

    def process_image(self, input_image, mouse_pos):
        # 转换输入图像为torch tensor，确保使用float32类型
        img_tensor = torch.from_numpy(np.array(input_image)).float().to(self.device) / 255.0
        height, width = img_tensor.shape[:2]
        resolution = torch.tensor([width, height], device=self.device, dtype=torch.float32)
        aspect_ratio = width / height
        inv_aspect = torch.tensor([1.0 / aspect_ratio, 1.0], device=self.device, dtype=torch.float32)
        
        # 创建输出图像tensor
        output = img_tensor.clone()
        
        # 设置折叠位置和方向
        fold_position = torch.tensor(mouse_pos, device=self.device, dtype=torch.float32)
        corner_position = torch.tensor([width, 0.0], device=self.device, dtype=torch.float32)
        
        # 计算折叠方向和角度
        dir_vector = fold_position - corner_position
        dir_vector = dir_vector / (torch.norm(dir_vector) + self.EPSILON)
        
        # 创建网格坐标
        y, x = torch.meshgrid(torch.arange(height, device=self.device, dtype=torch.float32), 
                            torch.arange(width, device=self.device, dtype=torch.float32),
                            indexing='ij')
        uv = torch.stack([x, y], dim=-1) / resolution
        uv[..., 0] *= aspect_ratio
        
        normalized_fold_pos = fold_position * torch.tensor([aspect_ratio, 1.0], 
                                                         device=self.device, 
                                                         dtype=torch.float32) / resolution
        
        # 计算卷曲效果
        origin = normalized_fold_pos - dir_vector * (
            (normalized_fold_pos[0] - aspect_ratio * self.if_greater_than(dir_vector[0], -self.EPSILON)) / 
            (dir_vector[0] + self.EPSILON)
        )
        
        # 批量计算距离和投影
        diff = origin.unsqueeze(0).unsqueeze(0) - uv
        proj = (diff * dir_vector).sum(dim=-1)
        curl_dist = torch.norm(normalized_fold_pos - origin)
        dist = proj - curl_dist
        
        # 计算线上点
        line_point = uv + dist.unsqueeze(-1) * dir_vector
        
        # 创建掩码
        mask_outside = dist > self.FOLD_RADIUS
        mask_curl = (dist > self.EPSILON) & ~mask_outside
        
        # 处理卷曲区域
        theta = torch.arcsin(torch.clamp(dist[mask_curl] / self.FOLD_RADIUS, -1, 1))
        p1 = line_point[mask_curl] - dir_vector * theta.unsqueeze(-1) * self.FOLD_RADIUS
        p2 = line_point[mask_curl] - dir_vector * (self.PI - theta).unsqueeze(-1) * self.FOLD_RADIUS
        
        # 计算采样坐标
        p1_coords = (p1 * inv_aspect * resolution).long()
        p2_coords = (p2 * inv_aspect * resolution).long()
        
        # 应用掩码和效果
        output[mask_outside] = 0
        
        # 处理有效的卷曲区域
        valid_coords = (p1_coords[:, 0] >= 0) & (p1_coords[:, 0] < width) & \
                      (p1_coords[:, 1] >= 0) & (p1_coords[:, 1] < height) & \
                      (p2_coords[:, 0] >= 0) & (p2_coords[:, 0] < width) & \
                      (p2_coords[:, 1] >= 0) & (p2_coords[:, 1] < height)
        
        p1_valid = p1_coords[valid_coords]
        p2_valid = p2_coords[valid_coords]
        curl_positions = torch.nonzero(mask_curl)[valid_coords]
        
        # 获取前后页面颜色
        # 修改获取前后页面颜色的部分
        front_color = img_tensor[p1_valid[:, 1], p1_valid[:, 0]]
        back_color = img_tensor[p2_valid[:, 1], p2_valid[:, 0]]
        
        # 光照效果
        light = ((self.FOLD_RADIUS - dist[mask_curl][valid_coords]) / self.FOLD_RADIUS) * 0.3
        light = light.float()
        
        # 修改：分别处理RGB和Alpha通道
        if img_tensor.shape[-1] == 4:  # 检查是否有Alpha通道
            # RGB通道添加光照效果
            back_color_rgb = torch.minimum(back_color[..., :3] + light.unsqueeze(-1), 
                                        torch.tensor(0.8, device=self.device, dtype=torch.float32))
            # 保持Alpha通道不变
            back_color = torch.cat([back_color_rgb, back_color[..., 3:]], dim=-1)
            
            # 混合颜色时考虑Alpha通道
            alpha_front = front_color[..., 3:4]
            alpha_back = back_color[..., 3:4]
            
            # 计算混合后的Alpha
            mixed_alpha = torch.clamp(
                alpha_front * (1 - self.OPACITY) + alpha_back * self.OPACITY,
                0, 1
            )
            
            # 计算混合后的RGB
            mixed_rgb = torch.clamp(
                front_color[..., :3] * (1 - self.OPACITY) + back_color[..., :3] * self.OPACITY,
                0, 1
            )
            
            # 合并RGB和Alpha通道
            mixed_color = torch.cat([mixed_rgb, mixed_alpha], dim=-1)
        else:
            # 原来的处理方式（无Alpha通道）
            back_color = torch.minimum(back_color + light.unsqueeze(-1), 
                                     torch.tensor(0.8, device=self.device, dtype=torch.float32))
            mixed_color = torch.clamp(
                front_color * (1 - self.OPACITY) + back_color * self.OPACITY,
                0, 1
            )
        
        output[curl_positions[:, 0], curl_positions[:, 1]] = mixed_color.to(output.dtype)

        # 转换回CPU并转为PIL图像
        output_np = (output.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(output_np)

class PageTurn:
    def __init__(self, input_image, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化页面翻转效果
        
        Args:
            input_image: PIL.Image 对象
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.effect = PageTurnEffect(device=device)
        self.input_image = input_image
        self.width, self.height = self.input_image.size
    
    @staticmethod
    def ease_in_out_cubic(x):
        """缓动函数，使动画更自然"""
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def calculate_fold_position(self, progress):
        """
        计算指定进度的折叠位置
        
        Args:
            progress (float): 动画进度，范围 0.0 到 1.0
        """
        progress = max(0.0, min(1.0, progress))
        start_pos = np.array([self.width, 0])
        end_pos = np.array([0, self.height])
        eased_progress = self.ease_in_out_cubic(progress)
        current_pos = start_pos + (end_pos - start_pos) * eased_progress
        return current_pos

    def generate_frame(self, progress):
        """
        生成指定进度的翻页效果图片
        
        Args:
            progress (float): 动画进度，范围 0.0 到 1.0
            
        Returns:
            PIL.Image: 处理后的图片
        """
        fold_pos = self.calculate_fold_position(progress)
        return self.effect.process_image(self.input_image, fold_pos)

    def generate_animation_frames(self, num_frames=60):
        """
        生成完整翻页动画的所有帧
        
        Args:
            num_frames (int): 动画帧数
            
        Returns:
            list: PIL.Image 对象列表
        """
        frames = []
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            frame = self.generate_frame(progress)
            frames.append(frame)
        return frames

def main():
    # 使用自定义图片
    input_image = Image.open('input.png').convert('RGBA')
    
    # 选择计算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建页面翻转效果实例
    page_turn = PageTurn(input_image, device=device)
    
    # 生成单帧
    progress = 0.55  # 设置进度
    result_image = page_turn.generate_frame(progress)
    
    # 保存结果
    result_image.save('output_frame.png')
    
    # 生成完整动画帧

    return result_image


# 直接运行代码并返回结果
result = main()
result  # 在Jupyter notebook中显示结果图片
