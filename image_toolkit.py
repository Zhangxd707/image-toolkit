import numpy as np
import math
from PIL import Image
import argparse

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert color image to grayscale using Pillow."""
    return image.convert('L')

def adjust_brightness(pixel_array: np.ndarray, brightness_factor: float) -> np.ndarray:
    """Adjust image brightness. Factor > 1 increases brightness."""
    adjusted = pixel_array * brightness_factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def otsu_thresholding(pixel_array: np.ndarray) -> tuple:
    """Perform Otsu's thresholding for automatic binary conversion using between-class variance maximization."""
    # Handle color images by converting to grayscale first
    if len(pixel_array.shape) == 3:
        gray_array = np.dot(pixel_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray_array = pixel_array
    
    # Calculate histogram
    hist, bins = np.histogram(gray_array.ravel(), bins=256, range=[0, 256])
    hist = hist.astype(float)
    
    # Total number of pixels
    total_pixels = gray_array.size
    
    # Current sum of pixel values and mean
    sum_total = np.sum(np.arange(256) * hist)
    sum_background = 0
    
    # Variables to track maximum between-class variance
    max_variance = 0
    optimal_threshold = 0
    
    # Background pixel count
    weight_background = 0
    
    for t in range(256):
        # Update background weight
        weight_background += hist[t]
        if weight_background == 0:
            continue
            
        # Foreground weight
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        # Update background sum
        sum_background += t * hist[t]
        
        # Calculate means
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Update maximum variance and optimal threshold
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t
    
    # Apply threshold to original image (handle both grayscale and color)
    if len(pixel_array.shape) == 3:
        # For color images, convert to grayscale first, then apply threshold
        gray_for_threshold = np.dot(pixel_array[...,:3], [0.2989, 0.5870, 0.1140])
        binary_image = np.where(gray_for_threshold > optimal_threshold, 255, 0).astype(np.uint8)
    else:
        binary_image = np.where(pixel_array > optimal_threshold, 255, 0).astype(np.uint8)
    
    return binary_image, optimal_threshold

def median_filter(pixel_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter for noise reduction. Kernel size must be odd."""
    pad_size = kernel_size // 2
    padded = np.pad(pixel_array, pad_size, mode='edge')
    output = np.zeros_like(pixel_array)
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            region = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            output[i-pad_size, j-pad_size] = np.median(region)
    return output.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Simple Image Processing Toolkit")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("operation", choices=["gray", "bright", "otsu", "median"], 
                        help="Operation: gray/bright/otsu/median")
    parser.add_argument("--output", "-o", default="output.png", help="Output image path")
    parser.add_argument("--factor", "-f", type=float, default=1.5, help="Brightness factor")
    parser.add_argument("--kernel", "-k", type=int, default=3, help="Median filter kernel size")
    
    args = parser.parse_args()
    
    img = Image.open(args.input)
    pixels = np.array(img)
    
    if args.operation == "gray":
        result = convert_to_grayscale(img)
    elif args.operation == "bright":
        result = Image.fromarray(adjust_brightness(pixels, args.factor))
    elif args.operation == "otsu":
        binary, threshold = otsu_thresholding(pixels)
        print(f"Otsu threshold: {threshold:.2f}")
        result = Image.fromarray(binary)
    elif args.operation == "median":
        result = Image.fromarray(median_filter(pixels, args.kernel))
    
    result.save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()