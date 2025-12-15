import random
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


class HybridAugment:
    def __init__(self, img_size=640, hsv_gains=(0.5, 0.5, 0.5), fft_mask_ratio=0.1):
        """
        Args:
            img_size: 输出图像尺寸
            hsv_gains: HSV通道的扰动强度 (hue, saturation, value)
            fft_mask_ratio: 频域掩码的半径比例 (0~1)
        """
        self.img_size = img_size
        self.hsv_gains = hsv_gains
        self.fft_mask_ratio = fft_mask_ratio

    def __call__(self, images, targets):
        # 1. Mosaic增强 (空间域)
        if random.random() < 0.8:  # 80%概率启用Mosaic
            img, targets = self.mosaic_augment(images, targets)
        else:
            img = images[0] if isinstance(images, list) else images

        # 2. FFT频域滤波 (频率域)
        if random.random() < 0.5:  # 50%概率启用FFT
            img = self.fft_augment(img)

        # 3. HSV色彩扰动 (空间域)
        img = self.hsv_augment(img)

        return img, targets

    def mosaic_augment(self, images, targets):
        """ 4图Mosaic拼接增强 """
        mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
        yc, xc = self.img_size, self.img_size  # 中心点

        # 随机选择4张图像的拼接位置
        indices = [random.randint(0, len(images) - 1) for _ in range(3)]
        indices = [0] + indices  # 第一张为基准图

        for i, index in enumerate(indices):
            img = images[index]
            h, w = img.shape[:2]

            # 放置位置 (左上、右上、左下、右下)
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            # 填充图像块
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # 调整目标框坐标 (此处简化，实际需按Mosaic逻辑调整targets)
        return mosaic_img, targets

    def fft_augment(self, img):
        """ 频域随机滤波增强 """
        # 转换为灰度图并做FFT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)

        # 随机选择高通或低通滤波
        if random.random() < 0.5:
            # 高通滤波 (增强边缘/小缺陷)
            h, w = magnitude.shape
            cx, cy = h // 2, w // 2
            radius = int(min(h, w) * self.fft_mask_ratio)
            cv2.circle(magnitude, (cx, cy), radius, 0, -1)
        else:
            # 低通滤波 (模拟模糊)
            magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)

        # 逆变换回空间域
        fft_shift = magnitude * np.exp(1j * phase)
        img_recon = np.fft.ifft2(np.fft.ifftshift(fft_shift)).real
        img_recon = np.uint8(np.clip(img_recon, 0, 255))
        return cv2.cvtColor(img_recon, cv2.COLOR_GRAY2BGR)

    def hsv_augment(self, img):
        """ HSV色彩空间扰动 """
        r = np.random.uniform(-1, 1, 3) * self.hsv_gains + 1
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[..., 0] = np.clip(img_hsv[..., 0] * r[0], 0, 179)  # Hue
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255)  # Saturation
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255)  # Value
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

# 初始化增强模块
augmentor = HybridAugment(img_size=640, hsv_gains=(0.5, 0.3, 0.3), fft_mask_ratio=0.15)

# 模拟输入 (4张图像和对应的目标框)
images = [np.random.randint(0, 255, (640,640,3), dtype=np.uint8) for _ in range(4)]
targets = [torch.rand((10,5)) for _ in range(4)]  # 假设每张图有10个目标

# 执行增强
aug_img, aug_targets = augmentor(images, targets)

# 可视化结果
cv2.imwrite("augmented.jpg", aug_img)