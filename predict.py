import os
import warnings
import sys
import json
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from Transformation import Transformation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

print("\n" + "=" * 60 + "\n")


def get_files():
    model_path = ""
    class_names_path = ""
    if len(sys.argv) > 2:
        model_packege_path = sys.argv[2]
        if os.path.isdir(model_packege_path):
            model_path = os.path.join(
                model_packege_path,
                "plant_disease_model.h5")

            class_names_path = os.path.join(
                model_packege_path,
                "class_names.json")

        elif os.path.isfile(model_packege_path)\
                and model_packege_path.endswith(".zip"):

            with zipfile.ZipFile(model_packege_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_packege_path))

            model_path = os.path.join(
                os.path.dirname(model_packege_path),
                "plant_disease_model.h5")

            class_names_path = os.path.join(
                os.path.dirname(model_packege_path),
                "class_names.json")

    elif os.path.exists("apple_plant_disease_model_pakege"):
        print("Using default model package: apple_plant_disease_model_pakege")
        model_path = os.path.join(
            "apple_plant_disease_model_pakege",
            "plant_disease_model.h5")
        class_names_path = os.path.join(
            "apple_plant_disease_model_pakege",
            "class_names.json")
    else:
        print("ERROR: Specified model package path is invalid:",
              model_packege_path)
        sys.exit(1)
    return model_path, class_names_path


def predict(model_path, image_path, class_names_path):
    '''Predict the class of the image using the trained model.'''
    try:
        # -------------------------------------------------------
        # Load model and class names
        # -------------------------------------------------------
        model = tf.keras.models.load_model(model_path)
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------------------------------
        # Prediction
        # -------------------------------------------------------

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        # np.argsort(predictions)[-3:][::-1]

        # -------------------------------------------------------
        # Display results in terminal
        # -------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"Image: {image_path}")
        print("=" * 60)

        print(f"Predicted class: {class_names[predicted_class_idx]}")
        print(f"Confidence: {confidence * 100:.2f}%")

        # -------------------------------------------------------
        # Plotting results
        # -------------------------------------------------------

        transformation = Transformation(image_path)
        transformed_images = transformation.create_mask()
        original_image = Image.open(image_path)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, pad=10)
        axes[0].axis('off')
        axes[1].imshow(transformed_images[0])
        axes[1].set_title('Transformed Image', fontsize=14, pad=10)
        axes[1].axis('off')

        fig.suptitle(
            f'The predicted class is => {class_names[predicted_class_idx]}',
            y=0.03,    # low value places the title near the bottom
            va='bottom',
            ha='center',
            fontsize=16,
            color='black')
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.show()

        return
    except Exception as e:
        raise RuntimeError(e)


def main():
    try:
        if len(sys.argv) < 2:
            print(
                "Usage: python predict.py <image_path> "
                "[model_packege_dir_or_zip]\n"
                "  <image_path>                 "
                "Path to the image to classify (required)\n"
                "  [model_packege_dir_or_zip]   "
                "Optional: directory containing "
                "'plant_disease_model.h5' and 'class_names.json',\n"
                "\n"
                "Examples:\n"
                "  python predict.py ./image.jpg\n"
                "  python predict.py ./image.jpg model_packege/\n"
            )
            print("\n" + "=" * 60 + "\n")
            sys.exit(1)

        image_path = sys.argv[1]

        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)

        model_path, class_names_path = get_files()
        predict(model_path, image_path, class_names_path)
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
