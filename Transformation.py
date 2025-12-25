#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import transformation_utils


class Transformation:
    def __init__(self, src_image_path):
        """Initialize with a single image path."""
        if not os.path.exists(src_image_path):
            raise FileNotFoundError(f"Image not found: {src_image_path}")

        self.src_path = src_image_path
        self.img = cv2.imread(src_image_path)

        if self.img is None:
            raise ValueError(f"Could not read image from '{src_image_path}'")

        # Convert BGR to RGB for matplotlib display
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.filename = os.path.splitext(os.path.basename(src_image_path))[0]
        self.extension = os.path.splitext(src_image_path)[1]

        # Initialize transformation results
        self.gray_img = None
        self.blur_img = None
        self.mask_img = None
        self.edge_img = None
        self.roi_img = None
        self.pseudolandmark_img = None
        self.histogram_data = None

    def gaussian_blur_threshold(self, dst_path=None):
        """Apply Gaussian blur followed by binary thresholding."""
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(self.img, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        self.gray_img = gray

        # Apply binary threshold
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        self.blur_img = threshold

        if dst_path:
            cv2.imwrite(dst_path, threshold)

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

        # Convert to RGB for matplotlib
        self.mask_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        if dst_path:
            cv2.imwrite(dst_path, masked_img)

        return masked_img, mask

    def roi_objects(self, dst_path=None):
        """        Detect and highlight regions
        of interest (disease spots or leaf contours)."""
        # Get mask first
        _, mask = self.create_mask()

        # Find contours
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

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

        # Convert to RGB for matplotlib
        self.roi_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if dst_path:
            cv2.imwrite(dst_path, result)

        return result

    def pseudolandmarks(self, dst_path=None, num_landmarks=100):
        """Generate equidistant pseudo-landmarks along the leaf contour."""
        _, mask = self.create_mask()

        # Find contours
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

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

        x_coords = np.interp(indices,
                             cumulative_distances,
                             contour_points[:, 0])
        y_coords = np.interp(indices,
                             cumulative_distances,
                             contour_points[:, 1])

        landmarks = np.vstack((x_coords, y_coords)).T

        # Draw on image
        result = self.img.copy()
        for point in landmarks:
            cv2.circle(result,
                       (int(point[0]), int(point[1])),
                       3, (255, 0, 0), -1)

        # Convert to RGB for matplotlib
        self.pseudolandmark_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if dst_path:
            cv2.imwrite(dst_path, result)

        return result

    def edge_detection(self, dst_path=None):
        """Apply Canny edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Store grayscale edges for display
        self.edge_img = edges

        if dst_path:
            cv2.imwrite(dst_path, edges)

        return edges

    def color_histogram(self, dst_path=None):
        """Generate and return color histogram data."""
        # Calculate histogram for each channel
        colors = ('b', 'g', 'r')
        hist_data = []

        for i, color in enumerate(colors):
            histogram = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            hist_data.append((color, histogram))

        self.histogram_data = hist_data

        # Save histogram if path provided
        if dst_path:
            plt.figure(figsize=(10, 6))
            for color, histogram in hist_data:
                color_name = 'Blue' if color == 'b'\
                    else 'Green' if color == 'g' else 'Red'
                plt.plot(histogram, color=color, label=color_name)

            plt.title('Color Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 256])
            plt.savefig(dst_path)
            plt.close()
        return hist_data

    def display_all_transformations(self):
        """Display all transformations for a single image."""
        print("\nDisplaying all transformations for:",
              self.filename, self.extension)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Image Transformations: {self.filename}{self.extension}',
                     fontsize=16)

        # Flatten axes for easy indexing
        axes = axes.flatten()

        # Generate all transformations
        self.gaussian_blur_threshold()
        self.create_mask()
        self.roi_objects()
        self.pseudolandmarks()
        self.edge_detection()
        self.color_histogram()

        # Plot 1: Original Image
        axes[0].imshow(self.img_rgb)
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')

        # Plot 2: Gaussian Blur + Threshold
        axes[1].imshow(self.blur_img, cmap='gray')
        axes[1].set_title('2. Gaussian Blur + Threshold')
        axes[1].axis('off')

        # Plot 3: Masked Image
        axes[2].imshow(self.mask_img)
        axes[2].set_title('3. Masked Image')
        axes[2].axis('off')

        # Plot 4: ROI Objects
        axes[3].imshow(self.roi_img)
        axes[3].set_title('4. ROI Objects')
        axes[3].axis('off')

        # Plot 5: Pseudolandmarks
        axes[4].imshow(self.pseudolandmark_img)
        axes[4].set_title('5. Pseudolandmarks')
        axes[4].axis('off')

        # Plot 6: Edge Detection
        axes[5].imshow(self.edge_img, cmap='gray')
        axes[5].set_title('6. Edge Detection')
        axes[5].axis('off')

        # Plot 7: Grayscale Image
        axes[6].imshow(self.gray_img, cmap='gray')
        axes[6].set_title('7. Grayscale Image')
        axes[6].axis('off')

        # Plot 8: Color Histogram
        if self.histogram_data:
            for color, histogram in self.histogram_data:
                color_name = 'Blue' if color == 'b'\
                    else 'Green' if color == 'g' else 'Red'
                axes[7].plot(histogram, color=color, label=color_name)

            axes[7].set_title('8. Color Histogram')
            axes[7].set_xlabel('Pixel Value')
            axes[7].set_ylabel('Frequency')
            axes[7].legend()
            axes[7].grid(True, alpha=0.3)
            axes[7].set_xlim([0, 256])

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

        print("\nAll transformations displayed successfully!")

    def save_transformed_images(self, transformation,
                                base_file_path, output_dir=".",
                                src_base_dir=None):
        """Save augmented images to files preserving directory structure"""
        import os

        valid_augmentations = [
            'gray_img', 'blur_img', 'mask_img', 'edge_img',
            'roi_img', 'pseudolandmark_img', 'histogram_data'
        ]

        if transformation not in valid_augmentations:
            print(f"Invalid transformation type: {transformation}")
            raise ValueError(
                f"Augmentation must be one of {valid_augmentations}")

        # Get the relative path from src_base_dir if provided
        if src_base_dir:
            # Make paths absolute for comparison
            abs_base_file = os.path.abspath(base_file_path)
            abs_src_base = os.path.abspath(src_base_dir)

            # Get relative path from source base directory
            try:
                rel_path = os.path.relpath(abs_base_file, abs_src_base)
            except ValueError:
                # If paths are on different drives (Windows), use basename
                rel_path = os.path.basename(base_file_path)

            # Remove extension from relative path
            base_name = os.path.splitext(rel_path)[0]
        else:
            # Just use the filename without extension
            base_name = os.path.splitext(os.path.basename(base_file_path))[0]

        # Create the output directory structure
        output_path_dir = os.path.dirname(os.path.join(output_dir, base_name))
        if output_path_dir and not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)

        # Define transformation suffixes
        suffix_map = {
            'gray_img': '_Gray.jpg',
            'blur_img': '_Blur.jpg',
            'mask_img': '_Mask.jpg',
            'edge_img': '_Edge.jpg',
            'roi_img': '_roi.jpg',
            'pseudolandmark_img': '_pseudolandmark.jpg',
            'histogram_data': '_Histogram.png'
        }

        # Build output path
        output_path = f"{output_dir}/{base_name}{suffix_map[transformation]}"

        # Save the appropriate transformation
        if transformation == 'histogram_data'\
                and self.histogram_data is not None:
            # Save histogram as a plot
            plt.figure(figsize=(10, 6))
            for color, histogram in self.histogram_data:
                color_name = 'Blue' if color == 'b'\
                    else 'Green' if color == 'g' else 'Red'
                plt.plot(histogram, color=color, label=color_name)

            plt.title('Color Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 256])
            plt.savefig(output_path)
            plt.close()
            print(f"  ✓ Saved: {output_path}")

        else:
            # Get the image data
            img_data = getattr(self, transformation, None)
            if img_data is not None:
                cv2.imwrite(output_path, img_data)
                print(f"  ✓ Saved: {output_path}")
            else:
                print("  ✗ Warning: No ", transformation,
                      " transformation found for", base_file_path)


def process_directory(src_dir, dst_dir='.', transformations=None):
    """Process all images in a directory, preserving subdirectory structure."""
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return

    # Default to all transformations if none specified
    if transformations is None:
        transformations = [
            'gray_img', 'blur_img', 'mask_img', 'edge_img',
            'roi_img', 'pseudolandmark_img', 'histogram_data'
        ]

    # Find all image files recursively
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(image_extensions):
                full_path = os.path.join(root, file)
                image_files.append(full_path)

    if not image_files:
        print(f"No image files found in {src_dir}")
        return

    print(f"\nFound {len(image_files)} images to process")
    print(f"Saving transformations to: {dst_dir}")
    print(f"Transformations to apply: {', '.join(transformations)}\n")

    successful = 0
    failed = 0

    for i, image_path in enumerate(image_files, 1):
        try:
            # Get relative path for display
            rel_path = os.path.relpath(image_path, src_dir)
            print(f"[{i}/{len(image_files)}] Processing: {rel_path}")

            # Create transformation object
            transformation_obj = Transformation(image_path)

            # Apply and save each transformation
            for trans in transformations:
                # Call the appropriate method to generate the transformation
                if trans == 'gray_img' or trans == 'blur_img':
                    transformation_obj.gaussian_blur_threshold()
                elif trans == 'mask_img':
                    transformation_obj.create_mask()
                elif trans == 'edge_img':
                    transformation_obj.edge_detection()
                elif trans == 'roi_img':
                    transformation_obj.roi_objects()
                elif trans == 'pseudolandmark_img':
                    transformation_obj.pseudolandmarks()
                elif trans == 'histogram_data':
                    transformation_obj.color_histogram()

                # Save the transformation
                transformation_obj.save_transformed_images(
                    transformation=trans,
                    base_file_path=image_path,
                    output_dir=dst_dir,
                    src_base_dir=src_dir
                )

            print()
            successful += 1
        except Exception as e:
            print("Error processing ",
                  os.path.basename(image_path), ":", {str(e)}, "\n")
            failed += 1
            continue

    print(f"\n{'='*60}")
    print("✓ Batch processing complete!")
    print(f"  Successful: {successful}/{len(image_files)}")
    if failed > 0:
        print(f"  Failed: {failed}/{len(image_files)}")
    print(f"{'='*60}")


def main():
    args = transformation_utils.parse_arguments()

    # Case 1: Single image with --file only - display all transformations
    if args.file and not args.dst:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        try:
            transformation = Transformation(args.file)
            transformation.display_all_transformations()
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Case 2: Single image with --file and --dst - save all transformations
    elif args.file and args.dst:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        try:
            print(f"\nProcessing single image: {args.file}")
            print(f"Saving transformations to: {args.dst}\n")

            transformation = Transformation(args.file)

            # List of all transformations
            transformations = [
                'gray_img', 'blur_img', 'mask_img', 'edge_img',
                'roi_img', 'pseudolandmark_img', 'histogram_data'
            ]

            # Apply and save each transformation
            for trans in transformations:
                # Generate the transformation
                if trans == 'gray_img' or trans == 'blur_img':
                    transformation.gaussian_blur_threshold()
                elif trans == 'mask_img':
                    transformation.create_mask()
                elif trans == 'edge_img':
                    transformation.edge_detection()
                elif trans == 'roi_img':
                    transformation.roi_objects()
                elif trans == 'pseudolandmark_img':
                    transformation.pseudolandmarks()
                elif trans == 'histogram_data':
                    transformation.color_histogram()

                # Save the transformation
                transformation.save_transformed_images(
                    transformation=trans,
                    base_file_path=args.file,
                    output_dir=args.dst,
                    src_base_dir=None  # No base directory for single files
                )

            print("\n✓ All transformations saved successfully!")

        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Case 3: Directory with --src only - display error (need --dst)
    elif args.src and not args.dst:
        print("Error: --dst is required when using --src for batch processing")
        print("Usage: python script.py --src ",
              "<source_directory> --dst <destination_directory>")
        sys.exit(1)

    # Case 4: Directory processing with
    #  --src and --dst - save all transformations
    elif args.src and args.dst:
        process_directory(args.src, args.dst)

    # Case 5: Invalid arguments
    else:
        print("Error: Invalid arguments")
        print("\nUsage:")
        print("  Display transformations: python script.py",
              "--file <image_path>")
        print("  Save single image:       python script.py",
              "--file <image_path> --dst <output_dir>")
        print("  Batch process:           python script.py",
              "--src <source_dir> --dst <output_dir>")
        transformation_utils.parse_arguments(help=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
