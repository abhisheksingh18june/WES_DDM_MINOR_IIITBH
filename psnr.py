import cv2
import numpy as np

def psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions and channels.")
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    R = 255
    psnr_value = 10 * np.log10((R ** 2) / mse)
    return psnr_value

if __name__ == "__main__":
   