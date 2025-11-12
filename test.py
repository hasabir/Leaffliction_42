#!/usr/bin/env python3

import argparse
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Transformation:
    def __init__(self, src_image_path):
        """Initialize with a single image path."""
        if not os.path.exists(src_image_path):
            raise FileNotFoundError(f"Image not found: {src_image_path}")
        
        self.src_path = src_image_path
        self.img = cv2.imread(src_image_path)
        
        if self.img is None:
            raise ValueError(f"Could not read image from '{src_image_path}'")
        
        self.filename = os.path.splitext(os.path.basename(src_image_path))[0]
        self.extension = os.path.splitext(src_image_path)[1]

    def gaussian_blur_threshold(self, dst_path=None):
        """Apply Gaussian blur followed by binary thresholding."""
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(self.img, (5, 5), 0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        if dst_path:
            cv2.imwrite(dst_path, threshold)
        else:
            cv2.imshow('1. Gaussian Blur + Threshold', threshold)
            cv2.waitKey(0)
        
        return threshold

    def create_mask(self, dst_path=None):
        """Create a binary mask to isolate the leaf from background."""
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        # Define range for green colors (leaves)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        
        if dst_path:
            cv2.imwrite(dst_path, masked_img)
        else:
            cv2.imshow('2. Masked Image', masked_img)
            cv2.waitKey(0)
        
        return masked_img, mask

    def roi_objects(self, dst_path=None):
        """Detect and highlight regions of interest (disease spots or leaf contours)."""
        # Get mask first
        _, mask = self.create_mask()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw on copy of original
        result = self.img.copy()
        
        if contours:
            # Draw largest contour (the leaf)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 3)
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add label
            cv2.putText(result, 'Leaf ROI', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        if dst_path:
            cv2.imwrite(dst_path, result)
        else:
            cv2.imshow('3. ROI Objects', result)
            cv2.waitKey(0)
        
        return result

    def analyze_object(self, dst_path=None):
        """Analyze and display object measurements."""
        _, mask = self.create_mask()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = self.img.copy()
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw contour
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
            
            # Add measurements as text
            y_offset = 30
            measurements = [
                f'Area: {int(area)} px²',
                f'Perimeter: {int(perimeter)} px',
                f'Width: {w} px',
                f'Height: {h} px',
                f'Aspect Ratio: {w/h:.2f}'
            ]
            
            for i, text in enumerate(measurements):
                cv2.putText(result, text, (10, y_offset + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if dst_path:
            cv2.imwrite(dst_path, result)
        else:
            cv2.imshow('4. Analyze Object', result)
            cv2.waitKey(0)
        
        return result

    def pseudolandmarks(self, dst_path=None, num_landmarks=100):
        """Generate equidistant pseudo-landmarks along the leaf contour."""
        _, mask = self.create_mask()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("Warning: No contours found for pseudolandmarks")
            return self.img
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.reshape(-1, 2)
        
        # Calculate cumulative distances
        distances = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        # Interpolate equidistant points
        total_length = cumulative_distances[-1]
        indices = np.linspace(0, total_length, num_landmarks)
        
        x_coords = np.interp(indices, cumulative_distances, contour_points[:, 0])
        y_coords = np.interp(indices, cumulative_distances, contour_points[:, 1])
        
        landmarks = np.vstack((x_coords, y_coords)).T
        
        # Draw on image
        result = self.img.copy()
        for point in landmarks:
            cv2.circle(result, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
        
        if dst_path:
            cv2.imwrite(dst_path, result)
        else:
            cv2.imshow('5. Pseudolandmarks', result)
            cv2.waitKey(0)
        
        return result

    def edge_detection(self, dst_path=None):
        """Apply Canny edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert to BGR for consistency
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        if dst_path:
            cv2.imwrite(dst_path, edges_bgr)
        else:
            cv2.imshow('6. Edge Detection', edges)
            cv2.waitKey(0)
        
        return edges

    def color_histogram(self, dst_path=None):
        """Generate and display color histogram."""
        plt.figure(figsize=(10, 6))
        
        colors = ('b', 'g', 'r')
        color_names = ('Blue', 'Green', 'Red')
        
        for i, (color, name) in enumerate(zip(colors, color_names)):
            histogram = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            plt.plot(histogram, color=color, label=name)
        
        plt.title(f'Color Histogram - {self.filename}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 256])
        
        if dst_path:
            plt.savefig(dst_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def display_all_transformations(self):
        """Display all transformations for a single image."""
        print(f"\nDisplaying all transformations for: {self.filename}{self.extension}")
        print("Press any key to move to next transformation...\n")
        
        self.gaussian_blur_threshold()
        self.create_mask()
        self.roi_objects()
        self.analyze_object()
        self.pseudolandmarks()
        self.edge_detection()
        cv2.destroyAllWindows()
        self.color_histogram()
        self.edge_detection_and_histogram()
        print("\nAll transformations displayed successfully!")

    def save_all_transformations(self, dst_dir):
        """Save all transformations to destination directory."""
        os.makedirs(dst_dir, exist_ok=True)
        
        base_name = self.filename
        
        print(f"Processing: {self.filename}{self.extension}")
        
        # Save each transformation
        self.gaussian_blur_threshold(
            os.path.join(dst_dir, f"{base_name}_gaussian_threshold{self.extension}")
        )
        self.create_mask(
            os.path.join(dst_dir, f"{base_name}_masked{self.extension}")
        )
        self.roi_objects(
            os.path.join(dst_dir, f"{base_name}_roi{self.extension}")
        )
        self.analyze_object(
            os.path.join(dst_dir, f"{base_name}_analyzed{self.extension}")
        )
        self.pseudolandmarks(
            os.path.join(dst_dir, f"{base_name}_landmarks{self.extension}")
        )
        self.edge_detection(
            os.path.join(dst_dir, f"{base_name}_edges{self.extension}")
        )
        self.color_histogram(
            os.path.join(dst_dir, f"{base_name}_histogram.png")
        )
        
        print(f"  ✓ Saved 7 transformations to {dst_dir}")
    
    def edge_detection_and_histogram(self, dst_path=None):
        """Display edge detection and color histogram together using matplotlib."""
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Edge Detection
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert BGR to RGB for proper display in matplotlib
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Display original image
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display edge detection
        ax2.imshow(edges_rgb)
        ax2.set_title('Edge Detection')
        ax2.axis('off')
        
        # Display color histogram
        colors = ('b', 'g', 'r')
        color_names = ('Blue', 'Green', 'Red')
        
        for i, (color, name) in enumerate(zip(colors, color_names)):
            histogram = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            ax3.plot(histogram, color=color, label=name)
        
        ax3.set_title('Color Histogram')
        ax3.set_xlabel('Pixel Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 256])
        
        plt.tight_layout()
        
        if dst_path:
            plt.savefig(dst_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return edges



def process_directory(src_dir, dst_dir):
    """Process all images in a directory."""
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return
    
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(src_dir) if f.endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {src_dir}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Saving transformations to: {dst_dir}\n")
    
    for i, image_file in enumerate(image_files, 1):
        try:
            src_path = os.path.join(src_dir, image_file)
            transformation = Transformation(src_path)
            transformation.save_all_transformations(dst_dir)
            print(f"Progress: {i}/{len(image_files)}\n")
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}\n")
            continue
    
    print(f"✓ Batch processing complete! {len(image_files)} images processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply computer vision transformations to extract features from leaf images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Display transformations for a single image:
    %(prog)s path/to/image.jpg
  
  Batch process a directory:
    %(prog)s -src path/to/images/ -dst path/to/output/
  
  With mask option (currently applies to all):
    %(prog)s -src path/to/images/ -dst path/to/output/ -mask
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Path to a single image file (displays transformations)'
    )
    parser.add_argument(
        '-src',
        type=str,
        help='Source directory containing images'
    )
    parser.add_argument(
        '-dst',
        type=str,
        help='Destination directory to save transformations'
    )
    parser.add_argument(
        '-mask',
        action='store_true',
        help='Apply masking during transformation process'
    )
    
    args = parser.parse_args()
    
    # Case 1: Single image - display all transformations
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        try:
            transformation = Transformation(args.file)
            transformation.display_all_transformations()
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    # Case 2: Directory processing - save all transformations
    elif args.src and args.dst:
        process_directory(args.src, args.dst)
    
    # Case 3: Invalid arguments
    else:
        parser.print_help()
        print("\nError: You must provide either:")
        print("  - A single image file to display transformations")
        print("  - Both -src and -dst for batch processing")
        sys.exit(1)


if __name__ == "__main__":
    main()













