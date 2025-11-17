import os
import pickle
import sys

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import  models, layers
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU
BATCH_SIZE = 16
# BATCH_SIZE = 32
# IMAGE_SIZE = 128
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=30

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds



def train(data_dir):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE)
    
    class_names = dataset.class_names
    train_dataset, val_dataset, test_dataset = get_dataset_partitions_tf(dataset)
    
    # FIX: Remove BATCH_SIZE from input_shape - it should only have (height, width, channels)
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    
    resize_and_rescale = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255),  # Resizing is already done by image_dataset_from_directory
    ])
    
    n_classes = len(class_names)
    
    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    
    # FIX: Use None for batch dimension when building
    model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_dataset,
        # Remove batch_size here - it's already batched in the dataset
        validation_data=val_dataset,
        verbose=1,
        epochs=EPOCHS,  # Use the EPOCHS constant you defined
    )
    return history


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_directory>")
        sys.exit(1)

    data_directory = sys.argv[1]
    train(data_directory)


if __name__ == '__main__':
    main()