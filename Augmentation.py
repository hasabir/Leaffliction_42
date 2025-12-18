#!/usr/bin/env python3

import sys
import cv2
import imutils
import numpy as np
from skimage import transform as tf


class Augmentation:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(self.image_path)

        if self.img is None:
            raise ValueError(
                f"Image not found or unable to load: {image_path}")

        # Initialize augmented images as None
        self.rotated = None
        self.flipped = None
        self.skewed = None
        self.blurred = None
        self.contrasted = None
        self.illuminated = None

    def rotate(self, angle=45):
        """Rotate image by specified angle"""
        self.rotated = imutils.rotate(self.img, angle)
        return self.rotated
    
    def flip(self, direction='vertical'):
        """Flip image horizontally or vertically"""
        if direction not in ['horizontal', 'vertical']:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")
        
        # Use self.img instead of reloading the image
        if direction == 'horizontal':
            self.flipped = cv2.flip(self.img, 1)
        else:
            self.flipped = cv2.flip(self.img, 0)
        
        return self.flipped

    def skew(self, factor=0.2):
        """Apply skew transformation to image"""
        # Convert BGR to RGB for skimage
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        affine_tf = tf.AffineTransform(rotation=np.deg2rad(15), shear=factor)
        skewed_rgb = tf.warp(img_rgb, inverse_map=affine_tf)
        
        # Convert back to BGR and scale to 0-255 for OpenCV
        self.skewed = (skewed_rgb * 255).astype(np.uint8)
        self.skewed = cv2.cvtColor(self.skewed, cv2.COLOR_RGB2BGR)
        
        return self.skewed
    
    def blur(self, ksize=(15, 15)):
        """Apply Gaussian blur to image"""
        self.blurred = cv2.GaussianBlur(self.img, ksize, 0)
        return self.blurred
    
    def contrast(self, alpha=1.5, beta=0):
        """Adjust image contrast and brightness"""
        self.contrasted = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)
        return self.contrasted

    def illuminate(self, gamma=2.5):
        """Adjust image illumination using gamma correction"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.illuminated = cv2.LUT(self.img, table)
        return self.illuminated
    

    def save_augmented_images(self, augmentation, base_file_path, output_dir="."):
        """Save augmented images to files"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        valid_augmentations = ['rotate', 'flip', 'skew', 'blur', 'contrast', 'illuminate']
        if augmentation not in valid_augmentations:
            print(f"Invalid augmentation type: {augmentation}")
            raise ValueError(f"Augmentation must be one of {valid_augmentations}")
        
        # Get the base name without extension and preserve directory structure
        base_name = os.path.splitext(base_file_path)[0]
        base_name = base_name.split("/")[1:]
        base_name = "/".join(base_name)
        
        
        #         # Get the base name without extension and preserve directory structure
        # base_name = os.path.splitext(base_file_path)[0]
        # base_name = base_name.split("/")[-2:]
        # base_name = "/".join(base_name)
        # # base_name = base_file_path.lstrip("./")  # remove leading ./

        # Create the output directory structure if it doesn't exist
        output_path_dir = os.path.dirname(os.path.join(output_dir, base_name))
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)
        
        # Save the appropriate augmented image
        if self.rotated is not None and augmentation == 'rotate':
            output_path = f"{output_dir}/{base_name}_Rotate.jpg"
            print(f"Writing rotated image to {output_path}")
            cv2.imwrite(output_path, self.rotated)
        
        elif self.flipped is not None and augmentation == 'flip':
            output_path = f"{output_dir}/{base_name}_Flip.jpg"
            print(f"Writing rotated image to {output_path}")
            
            cv2.imwrite(output_path, self.flipped)
        
        elif self.skewed is not None and augmentation == 'skew':
            output_path = f"{output_dir}/{base_name}_Skew.jpg"
            print(f"Writing rotated image to {output_path}")
            
            cv2.imwrite(output_path, self.skewed)
        
        elif self.blurred is not None and augmentation == 'blur':
            output_path = f"{output_dir}/{base_name}_Blur.jpg"
            print(f"Writing rotated image to {output_path}")
            
            cv2.imwrite(output_path, self.blurred)
        
        elif self.contrasted is not None and augmentation == 'contrast':
            output_path = f"{output_dir}/{base_name}_Contrast.jpg"
            print(f"Writing rotated image to {output_path}")
            
            cv2.imwrite(output_path, self.contrasted)
        
        elif self.illuminated is not None and augmentation == 'illuminate':
            output_path = f"{output_dir}/{base_name}_Illuminate.jpg"
            print(f"Writing rotated image to {output_path}")
            
            cv2.imwrite(output_path, self.illuminated)
        
        else:
            print(f"Warning: No {augmentation} augmentation found for {base_file_path}")

    
    

    def show_images(self, target_height=300):
        """Display original and augmented images"""
        # Generate augmented images if they don't exist
        if self.rotated is None:
            self.rotate()
        if self.flipped is None:
            self.flip()
            print(self.flip.__name__)
        if self.skewed is None:
            self.skew()
        if self.blurred is None:
            self.blur()
        if self.contrasted is None:
            self.contrast()
        if self.illuminated is None:
            self.illuminate()
        # Resize all images to target height
        orig = imutils.resize(self.img, height=target_height)
        rotated = imutils.resize(self.rotated, height=target_height)
        fliped = imutils.resize(self.flipped, height=target_height)
        skewed = imutils.resize(self.skewed, height=target_height)
        blured = imutils.resize(self.blurred, height=target_height)
        contrasted = imutils.resize(self.contrasted, height=target_height)
        illuminated = imutils.resize(self.illuminated, height=target_height)
        
        # Create montage
        montage = cv2.hconcat([orig, rotated, fliped, skewed, blured, contrasted, illuminated])
        
        # Display
        cv2.imshow("Original | Rotated 45 | Flipped | Skewed | Blured | Contrasted | Illuminated", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Augmentation.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    augmenter = Augmentation(image_path)
    augmenter.show_images()