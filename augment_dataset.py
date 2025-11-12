
from Distribution import Distribution
from Augmentation import Augmentation
import sys
import os

def sort_images_by_type(file_list):
    sorted_dict = {}
    for file_path in file_list:
        image_type = file_path.split("/")[2]
        if image_type not in sorted_dict:
            sorted_dict[image_type] = []
        sorted_dict[image_type].append(file_path)
    return sorted_dict


def augment_dataset(file_list, directory_path):
    output_dir = "augmented_directory"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_by_type = sort_images_by_type(file_list)
    base_length = max([len(images) for images in image_by_type.values()])
    augmentation_list = ['rotate', 'flip', 'skew', 'blur', 'contrast', 'illuminate']
    
    for image_type, images in image_by_type.items():
        current_length = len(images)
        augmentations_needed = base_length - current_length
        
        print(f"Processing {image_type}: Has {current_length} images, needs {augmentations_needed} more to reach {base_length}")
        
        if augmentations_needed <= 0:
            print(f"{image_type} already has enough images. Skipping.")
            continue
        
        augmentor_index = 0
        image_index = 0
        
        # Loop through images until we create enough augmentations
        while augmentations_needed > 0 and image_index < len(images):
            image = images[image_index]
            augmentor = Augmentation(image)
            print(f"  Processing image {image_index + 1}/{len(images)}: {image}")
            
            # Apply augmentations to this image until we've created enough OR cycled through all 6
            augmentations_applied = 0
            while augmentations_needed > 0 and augmentations_applied < len(augmentation_list):
                augmentation = augmentation_list[augmentor_index]
                
                # Apply the augmentation
                if augmentation == "rotate":
                    augmentor.rotate()
                elif augmentation == 'flip':
                    augmentor.flip()
                elif augmentation == 'skew':
                    augmentor.skew()
                elif augmentation == 'blur':
                    augmentor.blur()
                elif augmentation == 'contrast':
                    augmentor.contrast()
                elif augmentation == 'illuminate':
                    augmentor.illuminate()
                
                # Save the augmented image
                augmentor.save_augmented_images(augmentation, base_file_path=image, output_dir=output_dir)
                
                augmentations_needed -= 1
                augmentations_applied += 1
                augmentor_index = (augmentor_index + 1) % len(augmentation_list)
            
            image_index += 1
        
        print(f"Completed {image_type}: Created {base_length - current_length} augmented images\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python augment_dataset.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    distribution = Distribution(directory_path)
    file_list = []
    file_list = distribution.fetch_images(directory_path, file_list)
    augment_dataset(file_list, directory_path)