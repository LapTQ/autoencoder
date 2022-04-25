from models.autoencoder import *
from utils.datasets import *
import numpy as np

# i_layer = IdentityBlock(filters=[4, 4, 3], f=2)
# c_layer = ConvolutionalBlock(filters=[4, 4, 3], f=2, s=2)
encoder = Encoder()
decoder = Decoder()
x = np.ones((4, 133, 1925, 3))
latent = encoder(x)
x_re = decoder(latent)
# print(x_re)

dataset = AddressDataset(
    img_dir='data/data_samples_2',
    target_size=(133, 1925),
    batch_size=4,
    invert_color=True
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    [str(path) for path in Path('data/data_samples_2').glob('*.png')]
)
train_dataset = train_dataset.map(
    lambda x: load_img(x, True, (133, 1925))
)
train_dataset = train_dataset.batch(4)

for xs in train_dataset.take(1):
    print(xs.shape)

#
# imgs = next(iter(dataset))
# print(imgs.shape)
# plt.figure(figsize=(20, 3))
# plt.imshow(imgs[0])
# plt.show()
















# a = 60
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+3
# a = (a-1)*2+7
# print(a)