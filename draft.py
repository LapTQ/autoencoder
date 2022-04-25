from models.autoencoder import *
from utils.datasets import *
import numpy as np
import matplotlib.pyplot as plt

# i_layer = IdentityBlock(filters=[4, 4, 3], f=2)
# c_layer = ConvolutionalBlock(filters=[4, 4, 3], f=2, s=2)
# encoder = Encoder()
# decoder = Decoder()
# x = np.ones((1, 133, 1925, 3))
# latent = encoder(x)
# x_re = decoder(latent)

dataset = AddressDataset(
    img_dir='data/data_samples_2',
    target_size=(133, 1925),
    batch_size=4,
    invert_color=True
)

imgs = next(iter(dataset))
print(imgs.shape)
plt.figure(figsize=(20, 3))
plt.imshow(imgs[0])
plt.show()
















# a = 60
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+3
# a = (a-1)*2+7
# print(a)