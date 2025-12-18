import os
# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import matplotlib.pyplot as plt
from augment_dataset import augment_dataset
import shutil
from keras.preprocessing.image import save_img
import zipfile
import json
# from keras import layers, models
import keras
from keras import layers, models

class Train():
    def __init__(self, dataset_path="./dataset/Apples", output_dir="./model_packege"):
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.CHANNELS = 3
        self.EPOCHS = 3
        
        self.class_names = None
        self.dataset = None
        self.n_classes = 0
        self.train_ds = None
        self.model = None
        self.val_ds = None
        self.evaluate_accuracy_file = "evaluation_accuracy.txt"
        self.model_save_path = "plant_disease_model.h5"
        self.class_names_path = "class_names.json"
        
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            self._check_dataset_path(dataset_path)
            self._load_dataset(dataset_path)
        except Exception as e:
            print(f"Error initializing Train class: {e}")
            raise
        # self.val_ds = None

    def train(self):
        train_ds, val_ds = self._split_dataset()
        
        augmented_train_ds = self._augment_dataset(
            os.path.join(os.path.dirname(self.output_dir),\
                "train_ds") , train_ds)
        
        augmented_train_ds = augmented_train_ds.cache().shuffle(1000).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"Training batches: {len(augmented_train_ds)}")
        print(f"Validation batches: {len(val_ds)}")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.n_classes}")
        # print(f"len ugmented dataset: {}")

        # Step 8: Build model
        input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS)  # ✅ Fixed: No batch size

        resize_and_rescale = tf.keras.Sequential([
            layers.Rescaling(1.0 / 255)
        ])

        self.model = models.Sequential([
            resize_and_rescale,
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 4
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Added dropout for regularization
            layers.Dense(self.n_classes, activation='softmax'),
        ])

        self.model.summary()

        # Step 9: Compile model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        # Step 10: Train model
        self.model.fit(
            augmented_train_ds,
            validation_data=val_ds,
            verbose=1,
            epochs=self.EPOCHS,
        )

    def evaluate(self):
        directory_path = self.output_dir
        scores = self.model.evaluate(self.val_ds, verbose=1)
        print(f"\nValidation Loss: {scores[0]:.4f}")
        print(f"Validation Accuracy: {scores[1]:.4f} ({scores[1]*100:.2f}%)")
        
        accuracy_path = os.path.join(directory_path, self.evaluate_accuracy_file)
        # Save accuracy to file
        with open(accuracy_path, "w") as f:
            f.write(f"Validation Accuracy: {scores[1]*100:.2f}%\n")
        print(f"Validation accuracy saved to '{accuracy_path}'")


    def save_model(self):
        
        directory_path = self.output_dir
        # Save the model in HDF5 format
        self.model.save(os.path.join(directory_path, self.model_save_path))
        print(f"Model saved as '{directory_path}/{self.model_save_path}'")
        self.model.save(os.path.join(directory_path, 'model.keras'))
        print(f"Model saved as '{directory_path}/model.keras'")
        
        # Save class names
        class_names_path = os.path.join(directory_path, self.class_names_path)
        with open(class_names_path, "w") as f:
            json.dump(self.class_names, f)
        print(f"Class names saved to '{class_names_path}'")
        
        # Save augmented dataset
        # augmented_data_path = os.path.join(directory_path, "augmented_train_dataset")
        # if os.path.exists(augmented_data_path):
        #     shutil.rmtree(augmented_data_path)
        # shutil.copytree("augmented_train_directory", augmented_data_path)
        # print(f"Augmented training dataset copied to '{augmented_data_path}'")
        
        # zip the model package
        shutil.make_archive(directory_path, 'zip', directory_path)
        print(f"Model package zipped as '{directory_path}.zip'")
        
        
        
        
        
        

    
    def _load_dataset(self, dataset_path="./dataset/Apples"):
        self.dataset = keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            seed=123,
            shuffle=True,
            image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )

        self.class_names = self.dataset.class_names
        self.n_classes = len(self.class_names)



    def _split_dataset(self):
        train_ds, val_ds = self._get_dataset_partitions_tf(self.dataset)
        self.val_ds = val_ds
        return train_ds, val_ds


    def _augment_dataset(self, target_dir_path, train_ds):

        if os.path.exists(target_dir_path):
            shutil.rmtree(target_dir_path)
        os.makedirs(target_dir_path, exist_ok=True)
        counters = {name: 0 for name in self.class_names}


        #!
        for batch_images, batch_labels in train_ds:
            for img, lbl in zip(batch_images, batch_labels):
                cls = self.class_names[int(lbl.numpy())]
                cls_dir = os.path.join(target_dir_path, cls)
                os.makedirs(cls_dir, exist_ok=True)
                
                fname = f"{cls}_{counters[cls]:05d}.jpg"
                save_path = os.path.join(cls_dir, fname)
                
                save_img(save_path, img.numpy())
                counters[cls] += 1

        print("Saved training images to:", target_dir_path)
        print("Per-class counts:", counters)

        #! Step 4: Augment ONLY the training folder
        augmented_dir = os.path.join(self.output_dir, "augmented_train_directory")
        
        # print(f"output_dir: {self.output_dir}, augmented_dir: {augmented_dir}")
        # import sys
        # sys.exit(1)
        augment_dataset(directory_path=target_dir_path, output_dir=augmented_dir)

        #remove target_dir_path
        shutil.rmtree(target_dir_path)
        
        # with zipfile.ZipFile(augmented_zip_file_path, 'r') as zip_ref:
        #     zip_ref.extractall('augmented_train_directory')

        # Step 5: Load augmented training data
        augmented_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            augmented_dir,
            seed=123,
            shuffle=True,
            image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )
        return augmented_train_ds


    def _get_dataset_partitions_tf(self, ds, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
        assert (train_split + val_split) == 1
        
        ds_size = len(ds)
        
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        
        train_ds = ds.take(train_size)    
        val_ds = ds.skip(train_size).take(val_size)
        
        return train_ds, val_ds

    def _check_dataset_path(self, dataset_path):
        if dataset_path is None:
            raise ValueError("Dataset not loaded. Please load the dataset before training.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")


def main():
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./dataset/Apples"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        print("No output directory specified. Using default './model_packege'")
        output_dir = "./model_packege"

    trainer = Train(dataset_path=dataset_path, output_dir=output_dir)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main()

