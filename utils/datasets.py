import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def load_img(path, invert_color, target_size=None):
    img_string = tf.io.read_file(path)
    img = tf.image.decode_png(img_string)

    if invert_color:
        img = 255 - img

    if target_size is not None:
        target_height, target_width = target_size

        # this function cast the output to be float [0., 255.]
        img = tf.image.resize_with_pad(
            img,
            target_height=target_height,
            target_width=target_width
        ).numpy()

        img = img.astype(np.uint8)

    return img

def process_img(img):
    img = img / 255.

    return img

class AddressDataset(keras.utils.Sequence):
    """Iterate over the data as Numpy array.
    Reference: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(self, img_dir, target_size, batch_size=None, invert_color=True):
        self.img_paths = [str(path) for path in Path(img_dir).glob('*.png')]
        self.target_size = target_size
        self.batch_size = batch_size
        self.invert_color = invert_color

    def __len__(self):
        return len(self.img_paths) // (self.batch_size if self.batch_size is not None else 1)

    def __getitem__(self, idx):
        """Return images in batch if batch_size is not None."""
        img_num = self.batch_size if self.batch_size is not None else 1
        i = img_num * idx
        x = np.empty(shape=(img_num,) + self.target_size + (3,),
                     dtype=np.float32)

        for j, path in enumerate(self.img_paths[i: i + img_num]):
            img = load_img(path, invert_color=self.invert_color, target_size=self.target_size)
            img = process_img(img)
            x[j] = img

        if self.batch_size is None:
            x = np.squeeze(x,  axis=0)

        return x, x

