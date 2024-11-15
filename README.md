# DCGAN

这是一个利用DCGAN网络生成人脸面部图像的项目
python 3.8
pytorch 1.13以下

目前使用算力云4090D迭代1000次


![image](https://github.com/user-attachments/assets/9a1766b2-dfac-4b54-be83-8ae6eccbe5e5)

![image](https://github.com/user-attachments/assets/fddbfb12-48b5-4234-9447-dd91f20145b3)

![loss_plot](https://github.com/user-attachments/assets/550c188f-0c82-414f-908b-11dbfb953a3b)


# GAN网络总结

# GAN网络基本概念

生成对抗网络(generative adversarial network ,GAN)。GAN网络由生成器（Generator）G和判别器（Discriminator）D组成。

![image](https://github.com/user-attachments/assets/e36c3c5d-87e2-403a-b281-54f1ca4bc2d1)

生成器

生成器模型可以是任意结构的神经网络，其输入是随机噪声（torch.randn），输出则是生成的样本。生成器的目标是使生成的样本尽可能接近真实样本的分布，以欺骗判别器。GAN网络的生成器Generator输入是随机噪声，目的是每次生成不同的图片。但如果完全随机，就不知道生成的图像有什么特征，结果就会不可控，因此通常从一个先验的随机分布产生噪声。常用的随机分布：1)高斯分布：连续变量中最广泛使用的概率分布；2)均匀分布：连续变量x的一种简单分布。引入随机噪声使得生成的图片具有多样性。
生成器G是一个生成图片的网络，可以采用多层感知机、卷积网络、自编码器等。它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。通过下图模型结构讲解生成器如何一步步将噪声生成一张图片

![image](https://github.com/user-attachments/assets/7eab0446-7262-49a5-b732-507ca79c290c)

1) 输入：100维的向量；
2) 经过两个全连接层Fc1和Fc2、一个Resize，将噪声向量放大，得到128个7×7大小的特征图；
3) 进行上采样，以扩大特征图，得到128个14×14大小的特征图；
4) 经过第一个卷积Conv1，得到64个14×14的特征图；
5) 进行上采样，以扩大特征图，得到64个28×28大小的特征图；
6) 经过第二个卷积Conv2，将输入的噪声Z逐渐转化为1×28×28的单通道图片输出，得到生成图像。
全连接层作用：维度变换，变为高维，方便将噪声向量放大。因为全连接层计算量稍大，后序改进的GAN移除全连接层；最后一层激活函数通常使用tanh()：既起到激活作用，又起到归一作用，将生成器的输出归一化至[-1,1]，作为判别器的输入。也使GAN的训练更稳定，收敛速度更快，生成质量确实更高。

判别器

判别器模型同样可以是任意结构的神经网络，其输入是真实样本或生成器生成的样本，输出是一个概率值，表示输入样本是真实样本的概率。判别器的目标是尽可能准确地判断输入样本是真实样本还是生成样本。判别器D的输入为真实图像和生成器生成的图像，其目的是将生成的图像从真实图像中尽可能的分辨出来。属于二分类问题，通过下图模型结构讲解判别器如何区分真假图片。
1) 输入：单通道图像，尺寸为28*28像素(非固定值根据实际情况修改即可)；
2) 输出：二分类，样本是真或假。

![image](https://github.com/user-attachments/assets/5ad697d6-2363-4cf1-830a-a2fdf111b6fa)

1) 输入：1×28×28像素的图像；
2) 经过第一个卷积conv1，得到64个26×26的特征图，然后进行最大池化pool1，得到64个13×13的特征图；
3) 经过第二个卷积conv2，得到128个11×11的特征图，然后进行最大池化pool2，得到128个5×5的特征图；
4) 通过Resize将多维输入一维化；
5) 再经过两个全连接层fc1和fc2，得到原始图像的向量表达；
6) 最后通过Sigmoid激活函数，输出判别概率，图片是真是假的二分类。

GAN的损失函数

在训练过程中，生成器G（Generator）的目标就是尽量生成真实的图片去欺骗判别器D（Discriminator）。而D的目标就是尽量把G生成的图片和真实的图片区分开。这样，G和D构成了一个动态的“博弈过程”。在最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

![image](https://github.com/user-attachments/assets/04d595cc-14da-4639-a6e9-9fc6965a6792)

![image](https://github.com/user-attachments/assets/0d7b7805-ffef-446f-946b-386760eda65c)

编码器Encoder

Encoder目标是将输入序列编码成低维的向量表示或embedding，映射函数如下：
V→R^d

![image](https://github.com/user-attachments/assets/fc0df18e-db1d-4a19-99fe-9e5ef8400998)

Encoder一般是卷积神经网络，主要由卷积层，池化层和BatchNormalization层组成。卷积层负责获取图像局域特征，池化层对图像进行下采样并且将尺度不变特征传送到下一层，而BN主要对训练图像的分布归一化，加速学习。（Encoder网络结构不局限于卷积神经网络）
以人脸编码为例，Encoder将人脸图像压缩到短向量，这样短向量就包含了人脸图像的主要信息，例如该向量的元素可能表示人脸肤色、眉毛位置、眼睛大小等等。编码器学习不同人脸，那么它就能学习到人脸的共性：

![image](https://github.com/user-attachments/assets/80170899-a1ab-45ac-a6b5-d97443d63a50)

解码器Decoder

Decoder目标是利用Encoder输出的embedding，来解码关于图的结构信息。

![image](https://github.com/user-attachments/assets/dd4b59ec-c229-4f11-92d7-c43fdc4ffcf3)

输入是Node Pair的embeddings，输出是一个实数，衡量了这两个Node在中的相似性，映射关系如下：
R^d×R^d→R^+
Decoder对缩小后的特征图像向量进行上采样，然后对上采样后的图像进行卷积处理，目的是完善物体的几何形状，弥补Encoder当中池化层将物体缩小造成的细节损失。
以人脸编码、解码为例，Encoder对人脸进行编码之后，再用解码器Decoder学习人脸的特性，即由短向量恢复到人脸图像，如下图所示：

![image](https://github.com/user-attachments/assets/6c627a9e-90bc-4427-85a5-e07e37df3659)

# DCGAN网络

深度卷积对抗网络(Deep Convolutional Generative Adversarial Network，DCGAN)

改进点

1) 所有的pooling层使用strided卷积(判别器)和fractional-strided卷积(生成器)进行替换；
2) 使用Batch Normalization；
3) 移除全连接的隐层，让网络可以更深；
4) 在生成器上，除了输出层使用Tanh外，其它所有层的激活函数都使用ReLU；
5) 判别器所有层的激活函数都使用LeakyReLU。

![image](https://github.com/user-attachments/assets/5338a6a5-ff96-4bd0-a169-4c87d356f6b9)

