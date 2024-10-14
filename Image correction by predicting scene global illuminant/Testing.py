'''
DESCRIPTION:
This script tests a pre-trained Color Constancy Convolutional Neural Network (CNN) on a set of images. It visualizes the original image, 
the corrected image using the ground truth illuminant, and the corrected image using the predicted illuminant. It also displays the difference between the two corrected images. 
The script selects a random image and compares the ground truth correction against the CNN model's prediction.

Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
9/27/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor
from preprocessing import load_image, resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
from Network import ColorConstancyCNN
import random
import scipy.io as sio

'''
DESCRIPTION:
Visualizes the original image, ground truth illuminant correction, model-predicted illuminant correction, and the difference between the two corrected images.

PARAMETERS:
- image_path: File path of the input image.
- groundtruth_illuminant: The ground truth illuminant for the image.
- model: The trained Color Constancy CNN model.
- device: The device (CPU or GPU) to perform computations on.

RETURNS:
- None
'''

def visualize_input_vs_output(image_path, groundtruth_illuminant, model, device):
    # Load the image and ground truth illuminant
    image = load_image(image_path)
    groundtruth_illuminant = torch.tensor(groundtruth_illuminant, dtype=torch.float32).to(device)

    # Preprocess the image: resize, extract patches
    resized_image = resize_image_to_multiple_of_32(image)
    patches = get_image_patches(resized_image)
    
    # Convert patches to tensor and process with model
    patches_tensor = torch.stack([torch.tensor(patch, dtype=torch.float32) for patch in patches]).to(device)
    patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # Convert to (batch_size, channels, height, width)

    # Get model predictions for each patch
    model.eval()
    with torch.no_grad():
        outputs = model(patches_tensor)  # Outputs for each patch

    # Average the predictions across all patches to get the final predicted illuminant
    predicted_illuminant = outputs.mean(dim=0)

    # Apply the ground truth and predicted illuminant to the image
    corrected_image_groundtruth = apply_illuminant_correction(image, groundtruth_illuminant.cpu().numpy())
    corrected_image_predicted = apply_illuminant_correction(image, predicted_illuminant.cpu().numpy())

    # Display the original image, corrected image (ground truth), corrected image (predicted), and the difference
    show_comparison(image, corrected_image_groundtruth, corrected_image_predicted)

# def apply_illuminant_correction(image, illuminant):
#     """
#     Apply the given illuminant to the image. Assumes illuminant is a [3] tensor (RGB).
#     This performs simple color scaling based on the illuminant.
#     """
#     # Normalize the image by the illuminant (divide by illuminant and then multiply by a reference white point)
#     corrected_image = image / illuminant
#     corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)  # Ensure values are in [0, 255]
#     return corrected_image



'''
DESCRIPTION:
Applies the given illuminant correction to the image by performing color scaling. This adjusts the image colors based on the provided illuminant vector.

PARAMETERS:
- image: The original input image (as a NumPy array).
- illuminant: A tensor or array of shape [3] representing the RGB illuminant.

RETURNS:
- corrected_image: The image with the illuminant correction applied, as a NumPy array.
'''
def apply_illuminant_correction(image, illuminant):
    
    epsilon = 1e-6  # Small value to prevent division by zero
    illuminant = illuminant + epsilon  # Add epsilon to avoid division by zero
    print(illuminant)
    # Normalize the illuminant to ensure its range is between 0 and 1
    illuminant = illuminant / np.max(illuminant)  # Normalize illuminant to max 1

    # Apply the illuminant correction by dividing the image by the illuminant
    corrected_image = image / illuminant

    # Clip the corrected image to ensure valid range and convert to correct type
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    return corrected_image

'''
DESCRIPTION:
Displays a side-by-side comparison of the original image, corrected image using ground truth, corrected image using predicted illuminant, and the difference between the two corrected images.

PARAMETERS:
- image: The original input image.
- corrected_groundtruth: The image corrected using the ground truth illuminant.
- corrected_predicted: The image corrected using the predicted illuminant.

RETURNS:
- None
'''

def show_comparison(original_image, corrected_groundtruth, corrected_predicted):
    """
    Display the original image, corrected image (ground truth), corrected image (predicted), and their difference.
    """
    # Display the results using matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(corrected_groundtruth.astype(np.uint8))
    axes[1].set_title("Corrected Image (Ground Truth)")
    axes[1].axis("off")

    axes[2].imshow(corrected_predicted.astype(np.uint8))
    axes[2].set_title("Corrected Image (Predicted)")
    axes[2].axis("off")

    # Show the difference (absolute difference)
    difference = np.abs(corrected_groundtruth.astype(np.float32) - corrected_predicted.astype(np.float32))
    axes[3].imshow(difference.astype(np.uint8))
    axes[3].set_title("Difference")
    axes[3].axis("off")

    plt.show()




'''
DESCRIPTION:
Main function to load images, ground truths, and the pre-trained model, and visualize the comparison between model predictions and ground truth.

PARAMETERS:
- None

RETURNS:
- None
'''
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model = ColorConstancyCNN().to(device)
    model.load_state_dict(torch.load("color_constancy_angular_cnn_2_fold_3.pth",weights_only=True))
    
    # Load all image paths and ground truth illuminants (replace with actual paths)
    canon_1d_path = r'C:\\Users\\kodur\\OneDrive - Northeastern University\\Desktop\\Dataset_final\\1D'
    canon_5d_path = r'C:\\Users\\kodur\\OneDrive - Northeastern University\\Desktop\\Dataset_final\\5D - part 1'
    groundtruth_path = r'C:\\Users\\kodur\\OneDrive - Northeastern University\\Desktop\\Advanced Perception\\Bianco dataset\\real_illum_568.mat'
    
    canon_1d_images = sorted([os.path.join(canon_1d_path, f) for f in os.listdir(canon_1d_path) if f.endswith('.tiff')])
    canon_5d_images = sorted([os.path.join(canon_5d_path, f) for f in os.listdir(canon_5d_path) if f.endswith('.tiff')])
    image_paths = canon_1d_images + canon_5d_images
    
    # Load ground truth illuminants
    groundtruth_data = sio.loadmat(groundtruth_path)
    groundtruth_illuminants = groundtruth_data['real_rgb']

    # Create a list of images and their corresponding ground truth illuminants
    image_gt_pairs = list(zip(image_paths, groundtruth_illuminants))
    
    # Select a random image and its corresponding ground truth
    random_image, random_groundtruth = random.choice(image_gt_pairs)
    random_image, random_groundtruth = image_gt_pairs[0]
    
    # Visualize the input vs output for the selected image
    visualize_input_vs_output(random_image, random_groundtruth, model, device)


if __name__ == "__main__":
    main()