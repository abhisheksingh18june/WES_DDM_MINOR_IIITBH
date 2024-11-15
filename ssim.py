import cv2
import numpy as np
import os
from sewar.full_ref import msssim

def compute_rgb_ssim(image1, image2):
    return msssim(image1, image2)

def real_images(image_directorypath):
    """Return a list of file paths of images in the real image directory."""
    real_images_list = []
    for filename in os.listdir(image_directorypath):
        image_path = os.path.join(image_directorypath, filename)
        real_images_list.append(image_path)
    return real_images_list  # Return after loop completes

def generated_images(image_directorypath):
    """Return a list of file paths of images in the generated image directory."""
    generated_images_list = []
    for filename in os.listdir(image_directorypath):
        image_path = os.path.join(image_directorypath, filename)
        generated_images_list.append(image_path)
    return generated_images_list  # Return after loop completes

def main():
    """Main function to calculate SSIM for corresponding images."""
    real_image_directorypath = 'resized'
    generated_image_directorypath = 'rgb'
    
    # Get list of image paths from both directories
    real_images_list = real_images(real_image_directorypath)
    generated_images_list = generated_images(generated_image_directorypath)
    
    # Ensure both lists have the same length
    if len(real_images_list) != len(generated_images_list):
        raise ValueError("The number of real and generated images must be the same.")
    
    # Calculate SSIM for each pair of images
    ssim_cum=0
    for real_image_path, generated_image_path in zip(real_images_list, generated_images_list):
        real_image = cv2.imread(real_image_path)
        generated_image = cv2.imread(generated_image_path)
        
        if real_image is None or generated_image is None:
            print(f"Error reading image(s): {real_image_path} or {generated_image_path}")
            continue
        
        try:
            ssim_value = compute_rgb_ssim(real_image, generated_image)
            ssim_cum+=ssim_value
            print(f"SSIM between {real_image_path} and {generated_image_path}: {ssim_value:.4f}")
            with open('ssim.txt', 'a') as f:
                f.write(f"SSIM between {real_image_path} and {generated_image_path}: {ssim_value:.4f}\n")
        except Exception as e:
            print(f"Error calculating SSIM for {real_image_path} and {generated_image_path}: {e}")
            
    print(f"Average SSIM: {ssim_cum/len(real_images_list):.4f}")
    with open('ssim.txt', 'a') as f:
        f.write(f"Average SSIM: {ssim_cum/len(real_images_list):.4f}\n")
if __name__ == "__main__":
    main()
