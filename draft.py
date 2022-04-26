from models.autoencoder import *
from utils.datasets import *
import numpy as np
import matplotlib.pyplot as plt

# i_layer = IdentityBlock(filters=[4, 4, 3], f=2)
# c_layer = ConvolutionalBlock(filters=[4, 4, 3], f=2, s=2)
# x = np.ones((4, 133, 1925, 3))
# x = np.ones((4, 69, 773, 3))
# encoder = Encoder()
# decoder = Decoder()
# latent = encoder(x)
# latent = np.ones((4, 2, 24, 1))
# x_re = decoder(latent)
# print(encoder.sequential.summary())
# print(decoder.sequential.summary())
# i_layer(x)
# for v in i_layer.variables:
#     print(v)


# # USE TF.DATA
train_dataset = get_tf_dataset(
    img_dir='data/data_samples_2',
    target_size=(69, 773),
    batch_size=4,
    invert_color=True,
    normalize=True
)


for imgs, _ in train_dataset.take(1):
    print(imgs.shape)
    for img in imgs:
        plt.imshow(img)
        plt.show()
        break


# # USE KERAS SEQUENCE
# dataset = AddressDataset(
#     img_dir='data/data_samples_2',
#     target_size=(69, 773),
#     batch_size=4,
#     invert_color=True
# )
#
# imgs = next(iter(dataset))
# print(imgs[0].shape)
# plt.figure(figsize=(40, 3))
# plt.imshow(imgs[0][0])
# plt.axis('off')
# plt.tight_layout()
# plt.show()
















# a = 60
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+3
# a = (a-1)*2+7
# print(a)