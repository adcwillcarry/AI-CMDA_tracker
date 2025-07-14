# AI-CMDA_tracker
Before using this script:

Manually annotate cell division nodes and save the YOLO-format label files in the division_gt directory.
Train your own YOLO model weights for both BF (Bright Field) and FL (Fluorescence) images using your annotated data. Save the trained weights (e.g. best.pt) for use in the prediction step below.

(Optional)preprocess_images.py provides a batch preprocessing tool to convert 16-bit or 8-bit grayscale images into 3-channel 8-bit BGR images. It supports a variety of common image formats (such as PNG, JPG, TIFF, BMP) and saves the results to a specified folder, making them suitable for deep learning pipelines and visualization tasks.
python script.py --input_dir [source_folder] --output_dir [output_folder]

1. python yolo_predict.py \
  --model path/to/best.pt \
  --img_dir path/to/image_folder \
  --out_dir path/to/output_folder \
  --conf 0.5 \
  --save_img \
  --save_txt

2. python get_best_param.py \
  --bf_dir /path/to/BF/labels \
  --fl_dir /path/to/FL/labels \
  --gt_dir /path/to/division_gt \
  --out_dir ./output \
  --img_width 1224 \
  --img_height 904

3. python MultiModalDivNet.py \
    --bf_dir /path/to/BF/labels \
    --fl_dir /path/to/FL/labels \
    --param_json output/best_param.json \
    --output_dir output/nodes

4. python adaptive_CSRT.py \
    --node_dir ./division_nodes \
    --gfp_dir ./gfp_images \
    --fl_label_dir ./fl_labels \
    --out_dir ./output_tracks \
    --class_file ./division_nodes/classes.txt \
    --min_frame 1 \
    --save_vis

5. metabolism_analysis.m
Input Requirements
Image files: .tif format, consistent naming for both CFP/FRET and GFP channels.
YOLO label files: .txt files in standard format (class x_center y_center width height [normalized]).
classes.txt: class mapping file, produced in earlier steps.


Script Output
/analysis_labels/cellX.png — Ratio time series plot for each cell.
/analysis_labels/cellX.txt — Mean value per frame for each cell (and label type: 0=mother, 1=daughter).
/analysis/ — Summary plots across all cells.