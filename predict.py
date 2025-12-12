# import os
# import sys
# import zipfile
# import shutil
# import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2


# class PlantDiseasePredictor:
#     def __init__(self, model_path='plant_disease_model.h5', image_size=256):
#         """
#         Initialize the predictor with a trained model.
        
#         Args:
#             model_path: Path to the saved model (.h5 file)
#             image_size: Size to resize images to (default: 256)
#         """
#         self.image_size = image_size
#         self.model = None
#         self.class_names = None
        
#         # Load the model
#         self.load_model(model_path)
    
#     def load_model(self, model_path):
#         """Load the trained model from file."""
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file not found: {model_path}")
        
#         print(f"Loading model from: {model_path}")
#         self.model = keras.models.load_model(model_path)
#         print("Model loaded successfully!")
#         # Try to load class names from a file (JSON or plain text), or use default
#         class_names_file = 'class_names.json'
#         if os.path.exists(class_names_file):
#             try:
#                 with open(class_names_file, 'r') as f:
#                     data = json.load(f)
#                 # If file contains a JSON array of strings, use it
#                 if isinstance(data, list) and all(isinstance(x, str) for x in data):
#                     self.class_names = [x.strip() for x in data if x.strip()]
#                 else:
#                     # Fallback to reading as plain text lines
#                     with open(class_names_file, 'r') as f:
#                         self.class_names = [line.strip() for line in f.readlines() if line.strip()]
#             except Exception:
#                 # If JSON parsing fails, treat file as plain text lines
#                 with open(class_names_file, 'r') as f:
#                     self.class_names = [line.strip() for line in f.readlines() if line.strip()]
#         else:
#             # Default class names if file doesn't exist; guard access to model.output_shape
#             try:
#                 num_classes = int(self.model.output_shape[-1])
#             except Exception:
#                 num_classes = 1
#             self.class_names = [f"Class_{i}" for i in range(num_classes)]
#             self.class_names = [f"Class_{i}" for i in range(self.model.output_shape[-1])]
        
#         print(f"Classes: {self.class_names}")
    
#     def load_and_preprocess_image(self, image_path):
#         """
#         Load and preprocess an image for prediction.
        
#         Args:
#             image_path: Path to the image file
            
#         Returns:
#             Tuple of (original_image, preprocessed_image)
#         """
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image file not found: {image_path}")
        
#         # Load original image for display
#         original_image = Image.open(image_path)
#         original_array = np.array(original_image)
        
#         # Load and preprocess for model
#         img = keras.preprocessing.image.load_img(
#             image_path,
#             target_size=(self.image_size, self.image_size)
#         )
#         img_array = keras.preprocessing.image.img_to_array(img)
        
#         # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=0)
        
#         return original_array, img_array
    
#     def apply_transformations(self, image_path):
#         """
#         Apply various image transformations for visualization.
#         Similar to Part 3 of the project.
        
#         Args:
#             image_path: Path to the image file
            
#         Returns:
#             Dictionary of transformed images
#         """
#         # Read image
#         img = cv2.imread(image_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         transformations = {}
        
#         # Original
#         transformations['Original'] = img_rgb
        
#         # Gaussian Blur
#         transformations['Gaussian Blur'] = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        
#         # Grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         transformations['Grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
#         # Edge Detection
#         edges = cv2.Canny(gray, 100, 200)
#         transformations['Edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
#         # HSV Color Space
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         transformations['HSV'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
#         # Thresholding
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         transformations['Threshold'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
#         return transformations
    
#     def predict(self, image_path, display=True):
#         """
#         Make a prediction on an image.
        
#         Args:
#             image_path: Path to the image file
#             display: Whether to display the results (default: True)
            
#         Returns:
#             Tuple of (predicted_class, confidence, all_predictions)
#         """
#         print(f"\nPredicting disease for: {image_path}")
#         print("-" * 60)
        
#         # Load and preprocess image
#         original_image, preprocessed_image = self.load_and_preprocess_image(image_path)
        
#         # Make prediction
#         predictions = self.model.predict(preprocessed_image, verbose=0)
#         predicted_class_idx = np.argmax(predictions[0])
#         confidence = predictions[0][predicted_class_idx]
#         predicted_class = self.class_names[predicted_class_idx]
        
#         # Print results
#         print(f"Predicted Disease: {predicted_class}")
#         print(f"Confidence: {confidence * 100:.2f}%")
#         print("\nAll predictions:")
#         for i, (class_name, prob) in enumerate(zip(self.class_names, predictions[0])):
#             print(f"  {class_name}: {prob * 100:.2f}%")
        
#         if display:
#             self.display_results(image_path, original_image, predicted_class, confidence, predictions[0])
        
#         return predicted_class, confidence, predictions[0]
    
#     def display_results(self, image_path, original_image, predicted_class, confidence, all_predictions):
#         """
#         Display the original image, transformed images, and prediction results.
        
#         Args:
#             image_path: Path to the image file
#             original_image: Original image array
#             predicted_class: Predicted class name
#             confidence: Confidence score
#             all_predictions: Array of all class predictions
#         """
#         # Get transformations
#         transformations = self.apply_transformations(image_path)
        
#         # Create figure with subplots
#         fig = plt.figure(figsize=(16, 10))
        
#         # Plot transformations (2 rows, 3 columns)
#         transform_list = list(transformations.items())
#         for idx, (name, img) in enumerate(transform_list[:6]):
#             ax = plt.subplot(3, 3, idx + 1)
#             plt.imshow(img)
#             plt.title(name, fontsize=12, fontweight='bold')
#             plt.axis('off')
        
#         # Plot prediction bar chart
#         ax = plt.subplot(3, 3, 7)
#         colors = ['green' if i == np.argmax(all_predictions) else 'gray' 
#                   for i in range(len(self.class_names))]
#         bars = ax.barh(self.class_names, all_predictions * 100, color=colors)
#         ax.set_xlabel('Confidence (%)', fontsize=10)
#         ax.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
#         ax.set_xlim([0, 100])
        
#         # Add percentage labels on bars
#         for bar, pred in zip(bars, all_predictions):
#             width = bar.get_width()
#             ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
#                    f'{pred*100:.1f}%', 
#                    ha='left', va='center', fontsize=9)
        
#         # Plot final result text
#         ax = plt.subplot(3, 3, 8)
#         ax.axis('off')
#         result_text = f"""
#         PREDICTION RESULT
#         ═══════════════════
        
#         Disease: {predicted_class}
        
#         Confidence: {confidence * 100:.2f}%
        
#         Status: {'HIGH CONFIDENCE' if confidence > 0.9 else 'MODERATE' if confidence > 0.7 else 'LOW CONFIDENCE'}
#         """
#         ax.text(0.5, 0.5, result_text, 
#                ha='center', va='center', 
#                fontsize=12,
#                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
#                family='monospace')
        
#         # Add image info
#         ax = plt.subplot(3, 3, 9)
#         ax.axis('off')
#         info_text = f"""
#         IMAGE INFO
#         ═══════════════════
        
#         Path: {os.path.basename(image_path)}
        
#         Size: {original_image.shape[1]}x{original_image.shape[0]}
        
#         Model: CNN Classifier
#         """
#         ax.text(0.5, 0.5, info_text, 
#                ha='center', va='center', 
#                fontsize=10,
#                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
#                family='monospace')
        
#         plt.suptitle(f'Plant Disease Detection - {predicted_class}', 
#                     fontsize=16, fontweight='bold', y=0.98)
#         plt.tight_layout()
#         plt.show()


# def main():
#     """Main function to run predictions from command line."""
    
#     # Check arguments
#     if len(sys.argv) < 2:
#         print("Usage: python predict.py <image_path> [model_path]")
#         print("\nExample:")
#         print("  python predict.py ./Apple/apple_healthy/image (1).JPG")
#         print("  python predict.py ./test_image.jpg plant_disease_model.h5")
#         sys.exit(1)
    
#     image_path = sys.argv[1]
#     model_path = sys.argv[2] if len(sys.argv) > 2 else 'plant_disease_model.h5'
    
#     try:
#         # Create predictor
#         predictor = PlantDiseasePredictor(model_path=model_path)
        
#         # Make prediction
#         predicted_class, confidence, all_predictions = predictor.predict(image_path, display=True)
        
#         print("\n" + "="*60)
#         print("PREDICTION COMPLETE")
#         print("="*60)
        
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------------------------------
# Extract package
# -------------------------------------------------------
# with zipfile.ZipFile("leaffliction_package.zip", "r") as z:
#     z.extractall("leaffliction_package")




# -------------------------------------------------------
# Load model + class names
# -------------------------------------------------------
# model = tf.keras.models.load_model("./leaffliction_package/my_model.keras")
model = tf.keras.models.load_model("./plant_disease_model_3.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -------------------------------------------------------
# Validate CLI arguments
# -------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)

# -------------------------------------------------------
# Load + preprocess image (same as training)
# -------------------------------------------------------
img = Image.open(image_path)

# if img.mode != "RGB":
#     img = img.convert("RGB")

img = img.resize((256, 256))

img_array = np.array(img, dtype=np.float32)
# img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
predictions = model.predict(img_array, verbose=0)[0]

predicted_class_idx = np.argmax(predictions)
confidence = predictions[predicted_class_idx]

top3_idx = np.argsort(predictions)[-3:][::-1]

# -------------------------------------------------------
# Display results
# -------------------------------------------------------
print("\n" + "=" * 60)
print(f"Image: {image_path}")
print("=" * 60)

print(f"Predicted class: {class_names[predicted_class_idx]}")
print(f"Confidence: {confidence * 100:.2f}%")

print("\nTop 3 predictions:")
print("-" * 60)

for idx in top3_idx:
    print(f"  {class_names[idx]}: {predictions[idx] * 100:.2f}%")

print("=" * 60 + "\n")
