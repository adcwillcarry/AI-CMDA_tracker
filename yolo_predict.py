import argparse
from ultralytics import YOLO
import glob
import os

def main():
    """
    Main function for YOLO batch prediction script
    Processes all images in a directory using YOLO model for object detection
    """
    parser = argparse.ArgumentParser(description="YOLO batch image prediction script (requires 3-channel 8-bit image folder)")
    parser.add_argument('--model', type=str, required=True, help='Model weight path, e.g., best.pt')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images to predict (must be 8-bit 3-channel)')
    parser.add_argument('--out_dir', type=str, required=True, help='Prediction output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save_img', action='store_true', help='Whether to save images with bounding boxes')
    parser.add_argument('--save_txt', action='store_true', help='Whether to save txt label files')
    args = parser.parse_args()

    # Collect all image paths from the input directory
    # Supports multiple common image formats
    img_glob = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
        img_glob.extend(glob.glob(os.path.join(args.img_dir, ext)))
    
    # Check if any images were found
    if not img_glob:
        raise RuntimeError(f"Image directory is empty: {args.img_dir}")

    # Load YOLO model from the specified weights file
    model = YOLO(args.model)
    
    # Perform batch prediction on all collected images
    # The model will process all images in the list at once
    model.predict(
        source=img_glob,           # List of image paths to predict
        save=args.save_img,        # Save images with bounding boxes drawn
        save_txt=args.save_txt,    # Save prediction results as YOLO format txt files
        conf=args.conf,            # Confidence threshold for detections
        project=args.out_dir,      # Output project directory
        name='.'                   # Use current directory name for output subfolder
    )
    print(f"All image predictions completed, results saved in {args.out_dir}")

if __name__ == '__main__':
    main()
