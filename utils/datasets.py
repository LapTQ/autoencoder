import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def load_img(path):
    img_string = tf.io.read_file(path)
    img = tf.image.decode_png(img_string)

    return img

def process_img(img, invert_color, target_size=None, normalize=True):
    """
    Arguments:
        img: array-like image of type int
    if normalize is False, output is unit8, else float32.
    """
    if invert_color:
        img = 255 - img

    if target_size is not None:
        target_height, target_width = target_size

        # this function cast the output to be float [0., 255.]
        img = tf.image.resize_with_pad(
            img,
            target_height=target_height,
            target_width=target_width
        )#.numpy()

        # img = img.astype(np.uint8)
        img = tf.cast(img, tf.uint8)

    if normalize:
        img = img / 255

    return img

class AddressDataset(keras.utils.Sequence):
    """Iterate over the data as Numpy array.
    Reference: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(self, img_dir, target_size, batch_size=None, invert_color=True, normalize=True):
        self.img_paths = [str(path) for path in Path(img_dir).glob('*.png')]
        self.target_size = target_size
        self.batch_size = batch_size
        self.invert_color = invert_color
        self.normalize = normalize

    def __len__(self):
        return len(self.img_paths) // (self.batch_size if self.batch_size is not None else 1)

    def __getitem__(self, idx):
        """Return images in batch if batch_size is not None."""
        img_num = self.batch_size if self.batch_size is not None else 1
        i = img_num * idx
        x = np.empty(shape=(img_num,) + self.target_size + (3,),
                     dtype=np.float32)

        for j, path in enumerate(self.img_paths[i: i + img_num]):
            img = load_img(path)
            img = process_img(
                img,
                invert_color=self.invert_color,
                target_size=self.target_size,
                normalize=self.normalize
            )
            x[j] = img

        if self.batch_size is None:
            x = np.squeeze(x,  axis=0)

        return x, x

def get_tf_dataset(
        img_dir,
        target_size,
        batch_size=None,
        invert_color=True,
        normalize=True,
        shuffle=False,
        cache=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        [str(path) for path in Path(img_dir).glob('*.png')]
    )
    dataset = dataset.map(lambda x: load_img(x))
    dataset = dataset.map(
        lambda x: process_img(
            x,
            invert_color=invert_color,
            target_size=target_size,
            normalize=normalize
        )
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset