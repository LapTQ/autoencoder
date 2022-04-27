import argparse
from models.autoencoder import *
from utils.datasets import *
from tensorflow.keras import models, callbacks
import matplotlib.pyplot as plt

def run(
        pretrained,
        img,
        grayscale,
        invert_color,
        dilate,
        target_height,
        target_width,
        normalize,
        binarize,
        threshold,
):

    img = load_img(img)
    img = process_img(
        img,
        grayscale=grayscale,
        invert_color=invert_color,
        dilate=dilate,
        target_size=(target_height, target_width),
        normalize=normalize,
        binarize=binarize,
        threshold=threshold,
    )
    img = np.expand_dims(img, axis=0)

    autoencoder = models.load_model(pretrained)

    # autoencoder = Autoencoder()
    # autoencoder(img)
    # autoencoder.load_weights('checkpoints/checkpoint/variables.data-00000-of-00001')

    img_decoded = autoencoder(img)[0].numpy()

    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(img_decoded)
    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--pretrained', default='checkpoints/checkpoint', type=str)
    ap.add_argument('--img', default='data/private_test/0000_tests.png', type=str)
    ap.add_argument('--grayscale', default=True, type=bool)
    ap.add_argument('--invert-color', default=True, type=bool)
    ap.add_argument('--dilate', default=0, type=int)
    ap.add_argument('--target-height', default=133, type=int)
    ap.add_argument('--target-width', default=1925, type=int)
    ap.add_argument('--normalize', default=True, type=bool)
    ap.add_argument('--binarize', default=False, type=bool)
    ap.add_argument('--threshold', default=0.5, type=float)

    args = vars(ap.parse_args())

    run(**args)