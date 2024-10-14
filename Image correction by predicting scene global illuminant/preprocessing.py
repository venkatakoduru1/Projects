'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
9/27/2024
CS 7180 Advanced Perception
Travel Days Used: 1
DESCRIPTION:
This script preprocesses images by applying various techniques such as resizing images to dimensions that are multiples of 32, histogram stretching for contrast enhancement,
 and extracting non-overlapping patches from the images. It supports loading images in different formats, including DNG (Digital Negative), and processes them for further use. 
 The processed images are stored in a designated folder. This preprocessing is useful for preparing datasets for tasks like training neural networks or performing other image-based analyses.

'''
import rawpy
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
import os
import cv2
import scipy.io as sio

# Black levels for Canon 5D and Canon 1D cameras
# BLACK_LEVELS = {
#     'Canon 5D': 129,
#     'Canon 1D': 0
# }

# def remove_black_level_offset(image, camera_type='Canon 5D'):
#     """
#     Subtract the black level offset from the image based on the camera type.
#     :param image: The input image (in raw format or loaded).
#     :param camera_type: The type of camera used to capture the image (default 'Canon 5D').
#     """
#     black_level = BLACK_LEVELS.get(camera_type, 0)  # Default to 0 if unknown camera type
#     image = image.astype(np.float32)
#     image -= black_level
#     image = np.clip(image, 0, 255)  # Ensure the values stay within the valid range
#     return image.astype(np.uint8)


'''
DESCRIPTION:
Resizes the input image so that both dimensions are multiples of 32, while preserving the aspect ratio.

PARAMETERS:
- image: The input image to resize.
- max_size: The maximum size for the largest dimension of the image (default is 1200).

RETURNS:
- image: The resized image with dimensions as multiples of 32.
'''

def resize_image_to_multiple_of_32(image, max_size=1200):
    """Resize the image so that both dimensions are multiples of 32, while keeping aspect ratio."""
    #print("I am in the resize function")
    h, w = image.shape[:2]
    #print(f"Original image size: {w}x{h}")

    # Rescale the image so that the largest dimension is max_size
    if max(h, w) > max_size:
        scaling_factor = max_size / float(max(h, w))
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        #print(f"Resized image size (before adjusting to multiple of 32): {image.shape[1]}x{image.shape[0]}")

    # Adjust the dimensions to be multiples of 32
    new_h = (image.shape[0] // 32) * 32
    new_w = (image.shape[1] // 32) * 32

    # Resize the image again to ensure both dimensions are multiples of 32
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #print(f"Final image size (multiple of 32): {new_w}x{new_h}")
    
    return image


'''
DESCRIPTION:
Applies histogram stretching to the input image patch to enhance contrast.

PARAMETERS:
- image_patch: The input image patch on which to apply histogram stretching.

RETURNS:
- stretched_patch: The contrast-enhanced image patch.
'''
def histogram_stretching(image_patch):
    """Apply histogram stretching (contrast normalization) to a 32x32 image patch."""
    #print("I am in the histogram function")
    image_patch = image_patch.astype(np.float32)
    
    # For each channel (R, G, B), apply histogram stretching
    for i in range(3):  # Assuming the patch is in RGB format
        min_val = np.min(image_patch[:, :, i])
        max_val = np.max(image_patch[:, :, i])
        if max_val - min_val != 0:
            # Stretch the pixel values to the range [0, 255]
            image_patch[:, :, i] = (image_patch[:, :, i] - min_val) / (max_val - min_val) * 255
            
    return image_patch.astype(np.uint8)


'''
DESCRIPTION:
Processes the image and extracts non-overlapping patches of a given size.

PARAMETERS:
- image: The input image from which patches will be extracted.
- patch_size: The size of each patch to be extracted.

RETURNS:
- patches: A NumPy array containing the extracted image patches.
'''
def get_image_patches(image, patch_size=32):
    """Split the image into non-overlapping 32x32 patches."""
    #print("I am in the patch function")
    patches = []
    h, w, _ = image.shape
    #print(f"Extracting patches from image of size: {w}x{h}")

    # Iterate over the image to extract non-overlapping patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape == (patch_size, patch_size, 3):  # Ensure patch is of the correct size
                patches.append(patch)
    #print(f"Extracted {len(patches)} patches from the image.")
    return np.array(patches)


'''
DESCRIPTION:
Loads a DNG image file and converts it into an RGB image.

PARAMETERS:
- dng_path: The file path of the input DNG image.

RETURNS:
- rgb_image: The RGB-converted version of the input DNG image.
'''
def load_dng_image(dng_path):
    """Loads a DNG image and converts it to an RGB image."""
    #print("I am in the load function")
    try:
        raw = rawpy.imread(dng_path)
        rgb_image = raw.postprocess()  # Converts the raw DNG image to an RGB image
        #print(f"Successfully loaded and processed DNG file: {dng_path}")
        return rgb_image
    except Exception as e:
        print(f"Failed to load DNG image: {dng_path}. Error: {e}")
        return None

'''
DESCRIPTION:
Loads an image based on its file extension. It can handle both DNG (using rawpy) and other image formats (using OpenCV).

PARAMETERS:
- image_path: The file path of the input image.

RETURNS:
- image: The loaded image (either RGB or other formats) based on the file extension.
'''
def load_image(image_path):
    """Loads an image based on its file extension."""
    #print("I am in the load function")
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".dng":
        return load_dng_image(image_path)
    else:
        # Load other image formats (.jpg, .png, etc.) using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            pass
            #print(f"Successfully loaded image: {image_path}")
        else:
            print(f"Failed to load image: {image_path}")
        return image


folder_path_1D = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon1D'  # Folder containing your dataset of images
folder_path_5D = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon5D'  # Folder containing your dataset of images
output_folder = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\ProcessedDataset'  # Folder where processed patches will be saved
#process_images_in_folder(folder_path_1D,folder_path_5D, output_folder)
