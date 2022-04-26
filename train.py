import argparse
from models.autoencoder import *
from utils.datasets import *
from tensorflow.keras import callbacks

def run(
        epochs,
        batch_size,
        train_data,
        val_data,
        invert_color,
        target_height,
        target_width

):

    autoencoder = Autoencoder()

    train_dataset = AddressDataset(
        img_dir=train_data,
        target_size=(target_height, target_width),
        batch_size=batch_size,
        invert_color=invert_color
    )
    val_dataset = AddressDataset(
        img_dir=val_data,
        target_size=(target_height, target_width),
        batch_size=batch_size,
        invert_color=invert_color
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

    autoencoder.fit(
        train_dataset,
        epochs=epochs,
        shuffle=True,
        validation_data=val_dataset,
        callbacks=callback_list
    )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--epochs', default=24, type=int)
    ap.add_argument('--batch-size', default=32, type=int)
    ap.add_argument('--train-data', default='data/data_samples_2', type=str)
    ap.add_argument('--val-data', default='data/private_test', type=str)
    ap.add_argument('--invert-color', default=True, type=bool)
    ap.add_argument('--target-height', default=133, type=int)
    ap.add_argument('--target-width', default=1925, type=int)

    args = vars(ap.parse_args())

    run(**args)