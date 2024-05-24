import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class RainRemovalDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.rain_imgs_path = os.path.join(self.root_dir, 'rainy_image')
        self.rain_imgs = [f for f in os.listdir(self.rain_imgs_path) if f.endswith('.jpg') and not f.startswith('.')]
        self.clean_imgs_path = os.path.join(self.root_dir, 'ground_truth')
        self.num_rain_images_per_clean_image = 14  # 每个 clean_image 对应 14 个 rain_image

    def __len__(self):
        # 这里返回的是 rain_image 的总数
        return len(self.rain_imgs)

    def __getitem__(self, idx):
        rain_img_name = os.path.join(self.rain_imgs_path, self.rain_imgs[idx])  # 有雨点的照片

        # 获取对应的 clean_image 索引
        clean_img_idx = int(self.rain_imgs[idx].split('_')[0])
        clean_img_name = os.path.join(self.clean_imgs_path, f"{clean_img_idx}.jpg")  # 找出去除雨点对应的照片

        rain_image = Image.open(rain_img_name).convert('RGB')  # 打开雨点图片
        clean_image = Image.open(clean_img_name).convert('RGB')  # 打开对应的干净图片

        if self.transform:
            rain_image = self.transform(rain_image)
            clean_image = self.transform(clean_image)

        if self.train:
            # 如果是训练集，返回雨点图像和去雨点图像的数据
            return rain_image, clean_image
        else:
            # 如果是验证集，只返回雨点图像数据
            return rain_image
