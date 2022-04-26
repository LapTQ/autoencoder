import argparse
from models.autoencoder import *
from utils.datasets import *
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

def run(
        epochs,
        batch_size,
        train_data,
        val_data,
        invert_color,
        target_height,
        target_width,
        normalize,
        shuffle,
        cache
):

    autoencoder = Autoencoder()

    # train_dataset = AddressDataset(
    #     img_dir=train_data,
    #     target_size=(target_height, target_width),
    #     batch_size=batch_size,
    #     invert_color=invert_color,
    #     normalize=normalize
    # )
    # val_dataset = AddressDataset(
    #     img_dir=val_data,
    #     target_size=(target_height, target_width),
    #     batch_size=batch_size,
    #     invert_color=invert_color,
    #     normalize=normalize
    # )

    train_dataset = get_tf_dataset(
        img_dir=train_data,
        target_size=(target_height, target_width),
        batch_size=batch_size,
        invert_color=invert_color,
        normalize=normalize,
        shuffle=shuffle,
        cache=cache
    )
    val_dataset = get_tf_dataset(
        img_dir=val_data,
        target_size=(target_height, target_width),
        batch_size=batch_size,
        invert_color=invert_color,
        normalize=normalize,
        shuffle=False,
        cache=cache
    )

    autoencoder.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    callback_list = [
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath='checkpoints/checkpoint', save_best_only=True)
    ]

    history = autoencoder.fit(
        train_dataset,
        epochs=epochs,
        shuffle=True,
        validation_data=val_dataset,
        callbacks=callback_list
    )

    epoch_range = range(1, len(history.history['loss']) + 1)
    plt.plot(epoch_range, history.history['loss'], label='loss')
    plt.plot(epoch_range, history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--epochs', default=24, type=int)
    ap.add_argument('--batch-size', default=32, type=int)
    ap.add_argument('--train-data', default='data/data_samples_2', type=str)
    ap.add_argument('--val-data', default='data/private_test', type=str)
    ap.add_argument('--invert-color', default=True, type=bool)
    ap.add_argument('--target-height', default=69, type=int) #69 133
    ap.add_argument('--target-width', default=773, type=int) #773 1925
    ap.add_argument('--normalize', default=True, type=bool)
    ap.add_argument('--shuffle', default=False, type=bool)
    ap.add_argument('--cache', default=False, type=bool)

    args = vars(ap.parse_args())

    run(**args)