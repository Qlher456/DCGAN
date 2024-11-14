import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# 超参数
nz = 100  # 潜在向量大小
ngf = 64  # 生成器特征图大小
ndf = 64  # 判别器特征图大小
num_epochs = 1000
lr = 0.0001
beta1 = 0.5

# 创建Image文件夹
os.makedirs('Image', exist_ok=True)


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


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 LFW 数据集
    dataset = datasets.LFWPeople(root='data/', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 初始化模型
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 损失值列表
    lossesD = []
    lossesG = []

    # 训练GAN
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # 更新判别器
            ############################
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_images)
            lossD_real = criterion(output, label)
            lossD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ############################
            # 更新生成器
            ############################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()

            # 记录损失
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Step [{i}/{len(dataloader)}] '
                      f'Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}')

        # 生成并保存图像
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch + 1}")
        plt.imshow(grid.permute(1, 2, 0))

        # 保存图片到Image文件夹
        plt.savefig(f'Image/Epoch-{epoch + 1}.png')
        plt.close()

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(lossesG, label="G")
    plt.plot(lossesD, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Image/loss_plot.png')

    # 保存模型
    # 保存生成器
    torch.save(netG.state_dict(), 'dcgan_generator.pth')
    # 保存判别器
    torch.save(netD.state_dict(), 'dcgan_discriminator.pth')
