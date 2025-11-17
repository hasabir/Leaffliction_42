#!/usr/bin/env python3

import os
import sys
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


def train(data_dir):
    """
    Train a CNN model on plant disease images.
    
    Args:
        data_dir: Path to directory containing disease subdirectories
    """
    
    print("=" * 50)
    print("LEAFFLICTION - Training Plant Disease Classifier")
    print("=" * 50)
    
    # Verify directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found!")
        sys.exit(1)
    
    # Get number of classes
    categories = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(categories) == 0:
        print(f"Error: No subdirectories found in '{data_dir}'")
        sys.exit(1)
    
    num_classes = len(categories)
    print(f"\nFound {num_classes} disease categories:")
    for cat in categories:
        num_images = len(os.listdir(os.path.join(data_dir, cat)))
        print(f"  - {cat}: {num_images} images")
    
    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    
    print(f"\nTraining Configuration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Validation split: {VALIDATION_SPLIT * 100}%")
    
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixel values
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,           # Random rotation
        width_shift_range=0.2,       # Random horizontal shift
        height_shift_range=0.2,      # Random vertical shift
        horizontal_flip=True,        # Random horizontal flip
        zoom_range=0.2,              # Random zoom
        fill_mode='nearest'
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Check validation set size
    val_samples = val_generator.samples
    print(f"\nDataset Split:")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_samples}")
    
    if val_samples < 100:
        print(f"\n⚠️  WARNING: Validation set has only {val_samples} images!")
        print("   Project requires minimum 100 validation images.")
        print("   Consider using more data or adjusting split ratio.")
    
    # Build model using transfer learning (MobileNetV2)
    print("\nBuilding model with transfer learning (MobileNetV2)...")
    
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build complete model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nStarting training...")
    print("=" * 50)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    
    print(f"\n{'='*50}")
    print(f"FINAL VALIDATION ACCURACY: {val_accuracy * 100:.2f}%")
    print(f"{'='*50}")
    
    if val_accuracy >= 0.90:
        print("✅ SUCCESS! Achieved 90% / accuracy requirement!")
    else:
        print(f"⚠️  WARNING: Accuracy below 90% ({val_accuracy*100:.2f}%)")
        print("   Consider:")
        print("   - Training longer (increase epochs)")
        print("   - Using more augmented data")
        print("   - Fine-tuning the base model")
        print("   - Trying different architectures")
    
    # Generate predictions for confusion matrix
    print("\nGenerating validation predictions...")
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes
    
    # Classification report
    print("\nPer-Class Performance:")
    class_names = list(train_generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save training plots
    save_training_plots(history, cm, class_names)
    
    # Save model
    print("\nSaving model...")
    model.save('plant_disease_model.h5')
    print("✓ Model saved as 'plant_disease_model.h5'")
    
    # Save class mapping
    class_mapping = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("✓ Class mapping saved as 'class_mapping.json'")
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'final_val_accuracy': float(val_accuracy),
        'epochs_trained': len(history.history['accuracy']),
        'num_classes': num_classes,
        'validation_samples': int(val_samples)
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✓ Training history saved as 'training_history.json'")
    
    # Create zip package
    create_zip_package(data_dir)
    
    print("\n" + "="*50)
    print("Training complete! Package ready for evaluation.")
    print("="*50)


def save_training_plots(history, confusion_mat, class_names):
    """Save training history and confusion matrix plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training & Validation Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training & Validation Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # Plot 4: Summary Statistics
    axes[1, 1].axis('off')
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    epochs_trained = len(history.history['accuracy'])
    
    summary_text = f"""
    Training Summary
    ================
    
    Final Training Accuracy: {final_train_acc*100:.2f}%
    Final Validation Accuracy: {final_val_acc*100:.2f}%
    
    Epochs Trained: {epochs_trained}
    
    Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%
    (Epoch {np.argmax(history.history['val_accuracy']) + 1})
    
    Status: {'✅ PASSED (≥90%)' if final_val_acc >= 0.90 else '⚠️  Below 90%'}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("✓ Training plots saved as 'training_results.png'")
    plt.close()


def create_zip_package(data_dir):
    """Create zip package with model and dataset."""
    
    print("\nCreating ZIP package...")
    
    zip_filename = 'leaffliction_package.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model file
        if os.path.exists('plant_disease_model.h5'):
            zipf.write('plant_disease_model.h5')
            print("  ✓ Added model file")
        
        # Add best model if exists
        if os.path.exists('best_model.h5'):
            zipf.write('best_model.h5')
        
        # Add metadata files
        if os.path.exists('class_mapping.json'):
            zipf.write('class_mapping.json')
        if os.path.exists('training_history.json'):
            zipf.write('training_history.json')
        if os.path.exists('training_results.png'):
            zipf.write('training_results.png')
        
        print("  ✓ Added metadata files")
        
        # Add dataset (this might be large)
        print(f"  Adding dataset from {data_dir}...")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(data_dir))
                zipf.write(file_path, arcname)
        
        print("  ✓ Added dataset")
    
    print(f"\n✓ Package created: {zip_filename}")
    
    # Get file size
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    return zip_filename


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_directory>")
        print("\nExample:")
        print("  python train.py ./augmented_directory/Apple/")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    train(data_directory)


if __name__ == '__main__':
    main()