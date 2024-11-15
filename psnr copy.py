import cv2
import numpy as np
import os

def psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions and channels.")
    
    mse = np.mean((image1 - image2) ** 2)
    
    if mse == 0:
        return float('inf')  
    
    R = 255  
    psnr_value = 10 * np.log10((R ** 2) / mse)
    
    return psnr_value

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
    """Main function to calculate PSNR for corresponding images."""
    real_image_directorypath = 'resized'
    generated_image_directorypath = 'rgb'
    
    # Get list of image paths from both directories
    real_images_list = real_images(real_image_directorypath)
    generated_images_list = generated_images(generated_image_directorypath)
    
    # Ensure both lists have the same length
    if len(real_images_list) != len(generated_images_list):
        raise ValueError("The number of real and generated images must be the same.")
    
    # Calculate PSNR for each pair of images
    psnr_cum=0
    for real_image_path, generated_image_path in zip(real_images_list, generated_images_list):
        real_image = cv2.imread(real_image_path)
        generated_image = cv2.imread(generated_image_path)
        
        if real_image is None or generated_image is None:
            print(f"Error reading image(s): {real_image_path} or {generated_image_path}")
            continue
        
        try:
            psnr_value = psnr(real_image, generated_image)
            psnr_cum+=psnr_value
            # print(f"PSNR between {real_image_path} and {generated_image_path}: {psnr_value} dB")
            with open('psnr.txt', 'a') as f:
                f.write(f"PSNR between {real_image_path} and {generated_image_path}: {psnr_value} dB\n")
        except ValueError as e:
            print(f"Error calculating PSNR for {real_image_path} and {generated_image_path}: {e}")
            
    print(f"Average PSNR: {psnr_cum/len(real_images_list)} dB")
    with open('psnr.txt', 'a') as f:
        f.write(f"Average PSNR: {psnr_cum/len(real_images_list)} dB\n")
if __name__ == "__main__":
    main()
