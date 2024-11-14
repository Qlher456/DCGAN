import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# 超参数
nz = 100  # 潜在向量大小
ngf = 64  # 生成器特征图大小

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 加载生成器模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load('dcgan_generator.pth', map_location=device))
netG.eval()

# 生成随机噪声输入
fixed_noise = torch.randn(1, nz, 1, 1, device=device)

# 生成图像
with torch.no_grad():
    fake_image = netG(fixed_noise).detach().cpu()

# 显示并保存图像
plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(vutils.make_grid(fake_image, padding=2, normalize=True).permute(1, 2, 0))
# plt.savefig("generated_face.png")
plt.show()
