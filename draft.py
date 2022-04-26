from models.autoencoder2 import *
from utils.datasets import *
import numpy as np
import matplotlib.pyplot as plt

# i_layer = IdentityBlock(filters=[4, 4, 3], f=2)
# c_layer = ConvolutionalBlock(filters=[4, 4, 3], f=2, s=2)
encoder = Encoder()
decoder = Decoder()
x = np.ones((4, 133, 1925, 3))
x = np.ones((4, 69, 773, 3))
latent = encoder(x)
latent = np.ones((4, 2, 24, 1))
x_re = decoder(latent)
print(encoder.sequential.summary())
print(decoder.sequential.summary())


# USE TF.DATA
# train_dataset = tf.data.Dataset.from_tensor_slices(
#     [str(path) for path in Path('data/data_samples_2').glob('*.png')]
# )
# train_dataset = train_dataset.map(
#     lambda x: load_img(x, True, (133, 1925))
# )
# train_dataset = train_dataset.batch(4)
#
# for xs in train_dataset.take(1):
#     print(xs.shape)


# USE KERAS SEQUENCE
dataset = AddressDataset(
    img_dir='data/data_samples_2',
    target_size=(69, 773),
    batch_size=4,
    invert_color=True
)


# SHOW IMAGE BEFORE FEEDING
imgs = next(iter(dataset))
print(imgs[0].shape)
plt.figure(figsize=(40, 3))
plt.imshow(imgs[0][0])
plt.axis('off')
plt.tight_layout()
plt.show()
















# a = 60
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+3
# a = (a-1)*2+7
# print(a)