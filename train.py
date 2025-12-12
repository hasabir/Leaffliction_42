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
from tensorflow.keras.preprocessing.image import save_img
import zipfile
from keras import layers, models

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 30

# Step 1: Load ORIGINAL dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/Apples",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
n_classes = len(class_names)

# Step 2: Split FIRST
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    assert (train_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

train_ds, val_ds = get_dataset_partitions_tf(dataset)

# Step 3: Save training images to a separate folder
target_dir = "./Apples_train_only"

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

counters = {name: 0 for name in class_names}

for batch_images, batch_labels in train_ds:
    for img, lbl in zip(batch_images, batch_labels):
        cls = class_names[int(lbl.numpy())]
        cls_dir = os.path.join(target_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        fname = f"{cls}_{counters[cls]:05d}.jpg"
        save_path = os.path.join(cls_dir, fname)
        
        save_img(save_path, img.numpy())
        counters[cls] += 1

print("Saved training images to:", target_dir)
print("Per-class counts:", counters)

# Step 4: Augment ONLY the training folder
augmented_zip_file_path = augment_dataset(directory_path="./Apples_train_only")

with zipfile.ZipFile(augmented_zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('augmented_train_directory')

# Step 5: Load augmented training data
augmented_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "augmented_train_directory",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Step 6: Apply on-the-fly augmentation to training data
# data_augmentation = tf.keras.Sequential([
#     layers.RandomContrast(0.3),
#     layers.RandomBrightness(factor=0.3),
#     layers.RandomTranslation(0.1,0.1),
# ])

# augmented_train_ds = augmented_train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y),
#     num_parallel_calls=tf.data.AUTOTUNE
# )

# Step 7: Optimize pipelines (ONLY ONCE!)
augmented_train_ds = augmented_train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # No shuffle for validation!

print(f"Training batches: {len(augmented_train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Classes: {class_names}")
print(f"Number of classes: {n_classes}")
# print(f"len ugmented dataset: {}")

# Step 8: Build model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)  # ✅ Fixed: No batch size

resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1.0 / 255)
])

model = models.Sequential([
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
    layers.Dense(n_classes, activation='softmax'),
])

model.summary()

# Step 9: Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Step 10: Train model
history = model.fit(
    augmented_train_ds,  # ✅ Fixed: Use augmented dataset!
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,  # ✅ Fixed: Use EPOCHS constant
)

# Step 11: Evaluate
scores = model.evaluate(val_ds, verbose=1)
print(f"\nValidation Loss: {scores[0]:.4f}")
print(f"Validation Accuracy: {scores[1]:.4f} ({scores[1]*100:.2f}%)")

# Step 12: Save the model
model.save('plant_disease_model_3.h5')
print("Model saved as 'plant_disease_model_2.h5'")

# Step 13: Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved as 'training_history.png'")
model.save('my_model_3.keras')

# Now you can use augmented_train_ds and val_ds for training your model
#--------------------------------------------------------------------------

