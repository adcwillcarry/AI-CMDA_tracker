import os
import cv2
import glob
import argparse

def is_gray(img):
    """
    Check if the input image is grayscale
    Returns True if image has 1 or 2 dimensions, or 3 dimensions with only 1 channel
    """
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)

def convert_and_save(src_path, dst_path):
    """
    Convert various image formats to 8-bit 3-channel BGR format and save
    Handles 16-bit grayscale, 8-bit grayscale, and existing 3-channel images
    """
    # Read image with unchanged format to preserve original bit depth
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Cannot read image: {src_path}")
        return
    
    # Convert 16-bit grayscale to 8-bit 3-channel
    if len(img.shape) == 2 and img.dtype != 'uint8':
        # Normalize 16-bit values to 8-bit range (0-255)
        img8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
        # Convert grayscale to 3-channel BGR
        img3 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    # Convert 8-bit grayscale to 3-channel
    elif is_gray(img):
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Image is already 3-channel
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img3 = img
    else:
        print(f"Unsupported image channel format: {src_path}")
        return
    
    # Save the converted image
    cv2.imwrite(dst_path, img3)

def main():
    """
    Main function for batch image preprocessing
    Converts 16-bit grayscale or 8-bit grayscale images to 3-channel 8-bit images
    """
    parser = argparse.ArgumentParser(description='Batch convert 16-bit grayscale or 8-bit grayscale images to 3-channel 8-bit images')
    parser.add_argument('--input_dir', type=str, required=True, help='Source image folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output image folder')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect all image files from input directory
    # Support multiple common image formats
    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
        files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    print(f"Total {len(files)} images detected")
    
    # Process each image file
    for f in files:
        # Generate output path with same filename
        out_path = os.path.join(args.output_dir, os.path.basename(f))
        convert_and_save(f, out_path)
    
    print(f"Processing completed, all images saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
