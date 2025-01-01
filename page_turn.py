import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import math
import os

class PageTurnEffect:
    def __init__(self):
        # 常量定义
        self.PI = 3.14159265359
        self.FOLD_RADIUS = 0.2
        self.OPACITY = 0.5
        self.EPSILON = 1e-5

    def if_less_than(self, val, compare_val):
        return max(np.sign(compare_val - val), 0.0)

    def if_greater_than(self, val, compare_val):
        return max(np.sign(val - compare_val), 0.0)

    def process_image(self, input_image, mouse_pos):
        # 转换输入图像为numpy数组
        img_array = np.array(input_image).astype(np.float32) / 255.0
        height, width = img_array.shape[:2]
        resolution = np.array([width, height])
        aspect_ratio = width / height
        inv_aspect = np.array([1.0 / aspect_ratio, 1.0])
        
        # 创建输出图像数组
        output = img_array.copy()
        
        # 设置折叠位置和方向
        fold_position = np.array(mouse_pos)
        corner_position = np.array([width, 0.0])
        
        # 计算折叠方向和角度
        dir_vector = fold_position - corner_position
        dir_vector = dir_vector / (np.linalg.norm(dir_vector) + self.EPSILON)
        fold_angle = np.arctan2(dir_vector[1], dir_vector[0])

        # 处理每个像素
        for y in range(height):
            for x in range(width):
                uv = np.array([x, y]) / resolution
                uv[0] *= aspect_ratio
                
                normalized_fold_pos = fold_position * np.array([aspect_ratio, 1.0]) / resolution
                
                # 计算卷曲效果
                origin = normalized_fold_pos - dir_vector * (
                    (normalized_fold_pos[0] - aspect_ratio * self.if_greater_than(dir_vector[0], -self.EPSILON)) / 
                    (dir_vector[0] + self.EPSILON)
                )
                
                curl_dist = np.linalg.norm(normalized_fold_pos - origin)
                
                # 计算投影和距离
                proj = np.dot(origin - uv, dir_vector)
                dist = proj - curl_dist
                line_point = uv + dist * dir_vector
                
                # 根据距离应用不同的效果
                if dist > self.FOLD_RADIUS:
                    output[y, x] = [0, 0, 0, 0]
                        
                elif dist > self.EPSILON:
                    # 卷曲效果
                    theta = math.asin(min(dist / self.FOLD_RADIUS, 1.0))
                    p1 = line_point - dir_vector * theta * self.FOLD_RADIUS
                    p2 = line_point - dir_vector * (self.PI - theta) * self.FOLD_RADIUS
                    
                    p1_coords = (p1 * inv_aspect * resolution).astype(int)
                    p2_coords = (p2 * inv_aspect * resolution).astype(int)
                    
                    if (0 <= p1_coords[0] < width and 0 <= p1_coords[1] < height and
                        0 <= p2_coords[0] < width and 0 <= p2_coords[1] < height):
                        front_color = img_array[p1_coords[1], p1_coords[0]]
                        back_color = img_array[p2_coords[1], p2_coords[0]]
                        
                        # 光照效果
                        light = ((self.FOLD_RADIUS - dist) / self.FOLD_RADIUS) * 0.3
                        back_color = np.minimum(back_color + light, 1.0)
                        
                        # 混合颜色
                        output[y, x] = np.clip(
                            front_color * (1 - self.OPACITY) + back_color * self.OPACITY, 
                            0, 1
                        )
        
        return Image.fromarray((output * 255).astype(np.uint8))

class PageTurnAnimation:
    def __init__(self, image_path, output_path, fps=30, duration=2.0):
        self.image_path = image_path
        self.output_path = output_path
        self.fps = fps
        self.duration = duration
        self.total_frames = int(fps * duration)
        
        # 初始化效果处理器
        self.effect = PageTurnEffect()
        
        # 加载图片
        self.input_image = Image.open(image_path).convert('RGBA')
        self.width, self.height = self.input_image.size
        
    def calculate_fold_position(self, frame_idx):
        progress = frame_idx / self.total_frames
        
        # 定义折叠路径
        start_pos = np.array([self.width, 0])
        end_pos = np.array([0, self.height])
        
        # 使用缓动函数
        eased_progress = self.ease_in_out_cubic(progress)
        current_pos = start_pos + (end_pos - start_pos) * eased_progress
        
        return current_pos
    
    @staticmethod
    def ease_in_out_cubic(x):
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def generate_video(self):
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        try:
            for frame_idx in tqdm(range(self.total_frames), desc="生成视频帧"):
                # 计算当前帧的折叠位置
                fold_pos = self.calculate_fold_position(frame_idx)
                
                # 处理当前帧
                frame = self.effect.process_image(self.input_image, fold_pos)
                
                # 转换为OpenCV格式并写入
                frame_array = np.array(frame)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                video_writer.write(frame_array)
                
        finally:
            video_writer.release()

def main():
    # 设置输入输出路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input.jpg')
    output_path = os.path.join(current_dir, 'page_turn_animation.mp4')
    
    try:
        # 创建动画实例
        animation = PageTurnAnimation(
            image_path=input_path,
            output_path=output_path,
            fps=30,
            duration=2.0
        )
        
        # 生成视频
        animation.generate_video()
        print(f"视频已成功生成: {output_path}")
        
    except Exception as e:
        print(f"生成视频时出错: {str(e)}")

if __name__ == "__main__":
    main()
