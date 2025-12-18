
from Distribution import Distribution
from Augmentation import Augmentation
import sys
import os
import shutil
def sort_images_by_type(directory_path):
    sorted_dict = {}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}
    
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() not in image_exts:
                continue
            img_path = os.path.join(root, name)
            try:
                if not os.path.isfile(img_path):
                    continue
                # Use the immediate parent folder of the image as the category
                category = os.path.basename(root)
                sorted_dict.setdefault(category, []).append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    print("Image types found:", list(sorted_dict.keys()))
    return sorted_dict


def augment_dataset(directory_path, output_dir="augmented_directory"):
    
    directory_path = directory_path.strip("./")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_by_type = sort_images_by_type(directory_path)
    base_length = max([len(images) for images in image_by_type.values()])
    augmentation_list = ['rotate', 'flip', 'skew', 'blur', 'contrast', 'illuminate']
    
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # copy original images to output_dir
    shutil.copytree(directory_path, output_dir, dirs_exist_ok=True)
    
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
    # return output_dir
    # try:
    #     import zipfile
    #     print("Creating zip archive...")
    #     zip_filename = f"{output_dir}.zip"
    #     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #         for root, dirs, files in os.walk(output_dir):
    #             for file in files:
    #                 file_path = os.path.join(root, file)
    #                 arcname = os.path.relpath(file_path, start=output_dir)
    #                 zipf.write(file_path, arcname)
    #     print(f"Created zip archive: {zip_filename}")
    #     return zip_filename 
    # except Exception as e:
    #     print("zipfile module not found, cannot create zip archive.")


if __name__ == "__main__":
    if len(sys.argv) < 2 :
        print("Usage: python augment_dataset.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    distribution = Distribution(directory_path)
    if len(sys.argv) >=3:
        output_dir = sys.argv[2]
    else:
        output_dir = "augmented_directory"

    augment_dataset(directory_path, output_dir)    
    

# from Distribution import Distribution
# from Augmentation import Augmentation
# import sys
# import os
# import shutil
# def sort_images_by_type(directory_path):
#     sorted_dict = {}
#     image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}
    
#     for root, dirs, files in os.walk(directory_path):
#         for name in files:
#             _, ext = os.path.splitext(name)
#             if ext.lower() not in image_exts:
#                 continue
#             img_path = os.path.join(root, name)
#             try:
#                 if not os.path.isfile(img_path):
#                     continue
#                 # Use the immediate parent folder of the image as the category
#                 category = os.path.basename(root)
#                 sorted_dict.setdefault(category, []).append(img_path)
#             except Exception as e:
#                 print(f"Skipping {img_path}: {e}")
#     print("Image types found:", list(sorted_dict.keys()))
#     return sorted_dict


# def augment_dataset(directory_path, output_dir="augmented_directory"):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     image_by_type = sort_images_by_type(directory_path)
#     base_length = max([len(images) for images in image_by_type.values()])
#     augmentation_list = ['rotate', 'flip', 'skew', 'blur', 'contrast', 'illuminate']
    
#     # copy original images to output_dir
#     shutil.copytree(directory_path, output_dir, dirs_exist_ok=True)
    
#     for image_type, images in image_by_type.items():
#         current_length = len(images)
#         augmentations_needed = base_length - current_length
        
#         print(f"Processing {image_type}: Has {current_length} images, needs {augmentations_needed} more to reach {base_length}")
        
#         if augmentations_needed <= 0:
#             print(f"{image_type} already has enough images. Skipping.")
#             continue
        
#         augmentor_index = 0
#         image_index = 0
        
#         # Loop through images until we create enough augmentations
#         while augmentations_needed > 0 and image_index < len(images):
#             image = images[image_index]
#             augmentor = Augmentation(image)
#             print(f"  Processing image {image_index + 1}/{len(images)}: {image}")
            
#             # Apply augmentations to this image until we've created enough OR cycled through all 6
#             augmentations_applied = 0
#             while augmentations_needed > 0 and augmentations_applied < len(augmentation_list):
#                 augmentation = augmentation_list[augmentor_index]
                
#                 # Apply the augmentation
#                 if augmentation == "rotate":
#                     augmentor.rotate()
#                 elif augmentation == 'flip':
#                     augmentor.flip()
#                 elif augmentation == 'skew':
#                     augmentor.skew()
#                 elif augmentation == 'blur':
#                     augmentor.blur()
#                 elif augmentation == 'contrast':
#                     augmentor.contrast()
#                 elif augmentation == 'illuminate':
#                     augmentor.illuminate()
                
#                 # Save the augmented image
#                 augmentor.save_augmented_images(augmentation, base_file_path=image, output_dir=output_dir)
                
#                 augmentations_needed -= 1
#                 augmentations_applied += 1
#                 augmentor_index = (augmentor_index + 1) % len(augmentation_list)
            
#             image_index += 1
        
#         print(f"Completed {image_type}: Created {base_length - current_length} augmented images\n")
#         try:
#             import zipfile
            
#             zip_filename = f"{output_dir}.zip"
#             with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
#                 for root, dirs, files in os.walk(output_dir):
#                     for file in files:
#                         file_path = os.path.join(root, file)
#                         arcname = os.path.relpath(file_path, start=output_dir)
#                         zipf.write(file_path, arcname)
#             print(f"Created zip archive: {zip_filename}")
#             return zip_filename 
#         except Exception as e:
#             print("zipfile module not found, cannot create zip archive.")
#             return output_dir

