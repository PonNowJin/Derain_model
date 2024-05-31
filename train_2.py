import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RainRemovalDataset
import torch.nn.functional as F
from PIL import ImageEnhance

model_name = 'final_model_continue_residual.pth'
learning_rate = 1e-4
features = 32
epochs = 1


class GuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x, r):
        """Box filter implementation."""
        ch = x.shape[1]
        kernel_size = 2 * r + 1
        box_kernel = torch.ones((ch, 1, kernel_size, kernel_size), dtype=x.dtype, device=x.device)
        return F.conv2d(x, box_kernel, padding=r, groups=ch)

    def forward(self, I, p):
        r = self.radius
        eps = self.eps
        N = self.box_filter(torch.ones_like(I), r)

        mean_I = self.box_filter(I, r) / N
        mean_p = self.box_filter(p, r) / N
        mean_Ip = self.box_filter(I * p, r) / N

        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(I * I, r) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a, r) / N
        mean_b = self.box_filter(b, r) / N

        q = mean_a * I + mean_b
        return q


# 更改殘差連接，變成從頭連接到尾
class RainRemovalNet(nn.Module):
    def __init__(self, num_features=features, num_channels=3, kernel_size=3):
        super(RainRemovalNet, self).__init__()
        self.guided_filter = GuidedFilter(15, 1)
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

        layers = []
        num_residual_blocks = 16  # 图中表示为多层
        for _ in range(num_residual_blocks):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU())

        self.residual_layers = nn.Sequential(*layers)
        self.conv_final = nn.Conv2d(num_features, num_channels, kernel_size, padding=1)
        self.bn_final = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        base = self.guided_filter(x, x)  # 使用导向滤波器分解图像
        detail = x - base  # 获取细节层

        out = self.relu(self.bn1(self.conv1(detail)))  # 第一层卷积、批归一化和ReLU激活
        out_shortcut = out

        # 处理多个残差块
        for i in range(16):  # 16层残差块，注意这里是连续的16层
            out = self.residual_layers[3 * i](out)
            out = self.residual_layers[3 * i + 1](out)
            out = self.residual_layers[3 * i + 2](out)
            out_shortcut = out_shortcut + out  # 残差连接

        neg_residual = self.bn_final(self.conv_final(out_shortcut))  # 最后一层卷积和批归一化
        final_out = x + neg_residual  # 输出去雨后的图像

        return final_out

# 配置設備
device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 數據預處理
transform = transforms.Compose([
    transforms.Resize((384, 512)),
    # transforms.RandomResizedCrop((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加載數據集和數據加載器
train_dataset = RainRemovalDataset('/Users/ponfu/PycharmProject/IEEE_image_process/rainy_image_dataset/training', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = RainRemovalDataset('/Users/ponfu/PycharmProject/IEEE_image_process/rainy_image_dataset/testing', train=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 初始化網路和優化器
net = RainRemovalNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model_path = model_name
if os.path.exists(model_path):
    print("Load model successfully.")
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print("Train new model.")

# 訓練網路
num_epochs = epochs
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    torch.save(net.state_dict(), f'rain_removal_epoch_{epoch+1}.pth')

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_name)

print('Finished Training')

# 驗證網路
net.eval()
val_loss = 0.0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # 保存測試輸出的圖片
        for j in range(outputs.size(0)):
            output_image = outputs[j].cpu()
            output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            output_dir = 'Train_IEEE_output_train_2'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'image_{i * val_loader.batch_size + j + 1}.png')
            vutils.save_image(output_image, output_path)
            print(f'Saved: {output_path}')

val_loss /= len(val_loader)
print(f'Validation Loss: {val_loss:.4f}')
