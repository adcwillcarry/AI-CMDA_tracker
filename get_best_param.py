import pandas as pd
import numpy as np
import os
import itertools
import argparse
import json

def get_files(file_dir, file_type):
    """
    Get all files with specified file type from directory, excluding classes.txt
    """
    return sorted([f for f in os.listdir(file_dir) if f.endswith(file_type) and f != "classes.txt"])

def rect_from_yolo(label, imageWidth=1224, imageHeight=904):
    """
    Convert YOLO format (normalized) to rectangle coordinates (pixel)
    Returns [x_min, y_min, width, height]
    """
    x_center = label[1] * imageWidth
    y_center = label[2] * imageHeight
    width = label[3] * imageWidth
    height = label[4] * imageHeight
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    return [x_min, y_min, width, height]

def bounding_box_iou(rect1, rect2):
    """
    Calculate IoU and overlap ratios between two bounding boxes
    Returns (IoU, overlap_ratio1, overlap_ratio2)
    """
    xA = max(rect1[0], rect2[0])
    yA = max(rect1[1], rect2[1])
    xB = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    yB = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    if interArea == 0:
        return 0, 0, 0
    iou = interArea / (area1 + area2 - interArea)
    iou1 = interArea / area1
    iou2 = interArea / area2
    return iou, iou1, iou2

def load_yolo_txt(txt_path):
    """
    Load YOLO format labels from text file
    Returns empty list if file doesn't exist or is classes.txt
    """
    if not os.path.exists(txt_path) or txt_path.endswith("classes.txt"):
        return []
    arr = np.loadtxt(txt_path)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def get_next_name(fname):
    """
    Generate next frame filename by incrementing the frame number
    """
    prefix, num_txt = fname.rsplit('_', 1)
    num, ext = os.path.splitext(num_txt)
    num_new = f"{int(num)+1:03d}{ext}"
    return f"{prefix}_{num_new}"

def compute_division_bbox_IOU_area(gt_dir, img_width, img_height):
    """
    Compute IoU and area relationships between division cells from ground truth data
    Returns DataFrame with Parent-Daughter and Daughter-Daughter relationships
    """
    txt_files = get_files(gt_dir, '.txt')
    if 'classes.txt' in txt_files:
        txt_files.remove('classes.txt')
    
    # Load class labels
    with open(os.path.join(gt_dir, 'classes.txt')) as f:
        classes = np.array([int(x.strip()) for x in f.readlines()])
    
    # Load all labels from txt files
    labels = []
    for txt_file in txt_files:
        arr = np.loadtxt(os.path.join(gt_dir, txt_file))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        labels.extend(arr)
    labels = np.array(labels)
    
    # Replace class indices with actual class IDs
    for i in range(labels.shape[0]):
        labels[i, 0] = classes[int(labels[i, 0])]
    
    # Convert to rectangle coordinates
    rect_labels = np.zeros((labels.shape[0], 5))
    rect_labels[:, 0] = labels[:, 0]
    for i in range(labels.shape[0]):
        rect_labels[i, 1:5] = rect_from_yolo(labels[i, :], img_width, img_height)
    
    # Calculate IoU relationships between mother and daughter cells
    iou_relation = []
    for i in range(rect_labels.shape[0]):
        m_index = int(rect_labels[i, 0])
        d1_index = m_index * 10
        d2_index = m_index * 10 + 1
        m_label = rect_labels[rect_labels[:, 0] == m_index, 1:5]
        d1_label = rect_labels[rect_labels[:, 0] == d1_index, 1:5]
        d2_label = rect_labels[rect_labels[:, 0] == d2_index, 1:5]
        
        if len(d1_label) > 0 and len(d2_label) > 0:
            # Ensure d1 is the smaller daughter cell
            if d1_label[0, 2] * d1_label[0, 3] > d2_label[0, 2] * d2_label[0, 3]:
                d1_label, d2_label = d2_label, d1_label
            
            # Parent-Daughter relationships
            a, b, c = bounding_box_iou(m_label[0], d1_label[0])
            iou_relation.append([a, b, c, (d1_label[0, 2] * d1_label[0, 3]) / (m_label[0, 2] * m_label[0, 3]), 'Parent-Daughter'])
            a, b, c = bounding_box_iou(m_label[0], d2_label[0])
            iou_relation.append([a, b, c, (d2_label[0, 2] * d2_label[0, 3]) / (m_label[0, 2] * m_label[0, 3]), 'Parent-Daughter'])
            
            # Daughter-Daughter relationship
            a, b, c = bounding_box_iou(d1_label[0], d2_label[0])
            iou_relation.append([a, b, c, (d1_label[0, 2] * d1_label[0, 3]) / (d2_label[0, 2] * d2_label[0, 3]), 'Daughter-Daughter'])
    
    columns = ['P_IoU', 'P_M', 'P_D', 'Size_Ratio', 'Relation_Type']
    df = pd.DataFrame(iou_relation, columns=columns)
    return df

def main():
    """
    Main function for division node ablation parameter selection
    Automatically computes division_bbox_IOU_area and outputs best parameters as JSON
    """
    parser = argparse.ArgumentParser(description="Division node ablation parameter selection, outputs only best parameters JSON (automatically computes division_bbox_IOU_area)")
    parser.add_argument('--bf_dir', type=str, required=True, help='Bright field label directory')
    parser.add_argument('--fl_dir', type=str, required=True, help='Fluorescence label directory')
    parser.add_argument('--gt_dir', type=str, default='division_gt', help='Ground truth division labels directory')
    parser.add_argument('--out_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--img_width', type=int, default=1224, help='Image width')
    parser.add_argument('--img_height', type=int, default=904, help='Image height')
    args = parser.parse_args()

    bf_dir, fl_dir, gt_dir = args.bf_dir, args.fl_dir, args.gt_dir
    out_dir = args.out_dir
    img_width, img_height = args.img_width, args.img_height
    os.makedirs(out_dir, exist_ok=True)

    # --- Automatically generate division_bbox_IOU_area.csv dataframe ---
    df = compute_division_bbox_IOU_area(gt_dir, img_width, img_height)

    bf_txts = get_files(bf_dir, '.txt')
    
    def get_fl_name(bf_name):
        """
        Convert bright field filename to corresponding fluorescence filename
        """
        fl_name = bf_name.replace('_3_', '_2_').replace('Bright Field', 'GFP')
        prefix, num_txt = fl_name.rsplit('_', 1)
        num, ext = os.path.splitext(num_txt)
        num_new = f"{int(num)+1:03d}{ext}"
        return f"{prefix}_{num_new}"

    def get_next_gt_name(gt_name):
        """
        Get next frame ground truth filename
        """
        prefix, num_txt = gt_name.rsplit('_', 1)
        num, ext = os.path.splitext(num_txt)
        num_new = f"{int(num)+1:03d}{ext}"
        return f"{prefix}_{num_new}"

    # Define parameter names and calculate parameter ranges
    param_names = ['P_IoU', 'P_M', 'P_D', 'Size_Ratio']
    md = df[df['Relation_Type'] == 'Parent-Daughter']
    dd = df[df['Relation_Type'] == 'Daughter-Daughter']
    pm_limits = {k: [md[k].min(), md[k].max()] for k in param_names}
    dd_limits = {k: [dd[k].min(), dd[k].max()] for k in param_names}

    def load_yolo_txt(txt_path):
        """
        Local function to load YOLO format labels
        """
        if not os.path.exists(txt_path) or txt_path.endswith("classes.txt"):
            return []
        arr = np.loadtxt(txt_path)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def rect_from_yolo(label, imageWidth, imageHeight):
        """
        Local function to convert YOLO format to rectangle coordinates
        """
        x_center = label[1] * imageWidth
        y_center = label[2] * imageHeight
        width = label[3] * imageWidth
        height = label[4] * imageHeight
        x_min = round(x_center - width / 2)
        y_min = round(y_center - height / 2)
        return [x_min, y_min, width, height]

    def bounding_box_iou(rect1, rect2):
        """
        Local function to calculate IoU between two rectangles
        """
        xA = max(rect1[0], rect2[0])
        yA = max(rect1[1], rect2[1])
        xB = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        yB = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        area1 = rect1[2] * rect1[3]
        area2 = rect2[2] * rect2[3]
        if interArea == 0:
            return 0
        iou = interArea / (area1 + area2 - interArea)
        return iou

    # Generate all possible parameter switch combinations (at least one parameter must be enabled)
    switch_list = [s for s in itertools.product([0, 1], repeat=4) if sum(s) > 0]
    results = []
    
    # Test all parameter combinations
    for pm_switch in switch_list:
        for dd_switch in switch_list:
            hit_count = 0
            wrong_count = 0
            pm_used = dict(zip([f'pm_{k}' for k in param_names], pm_switch))
            dd_used = dict(zip([f'dd_{k}' for k in param_names], dd_switch))
            
            # Process each bright field image
            for bf_file in bf_txts:
                # Get file paths for current test case
                bf_path = os.path.join(bf_dir, bf_file)
                fl_file = get_fl_name(bf_file)
                fl_path = os.path.join(fl_dir, fl_file)
                gt_path = os.path.join(gt_dir, bf_file)
                next_gt_file = get_next_gt_name(bf_file)
                next_gt_path = os.path.join(gt_dir, next_gt_file)
                
                # Skip if any required file doesn't exist
                if not (os.path.exists(fl_path) and os.path.exists(gt_path) and os.path.exists(next_gt_path)):
                    continue
                    
                # Load cell data from different sources
                mother_cells = load_yolo_txt(bf_path)
                child_cells = load_yolo_txt(fl_path)
                gt_cells = load_yolo_txt(gt_path)
                next_gt_cells = load_yolo_txt(next_gt_path)
                
                # Test each mother cell for division detection
                for mother in mother_cells:
                    mother_rect = rect_from_yolo(mother, img_width, img_height)
                    child_candidates = []
                    
                    # Find child cells that meet parent-child criteria
                    for child in child_cells:
                        child_rect = rect_from_yolo(child, img_width, img_height)
                        area_m = mother_rect[2] * mother_rect[3]
                        area_c = child_rect[2] * child_rect[3]
                        
                        # Calculate intersection area
                        interW = max(0, min(mother_rect[0]+mother_rect[2], child_rect[0]+child_rect[2]) - max(mother_rect[0], child_rect[0]))
                        interH = max(0, min(mother_rect[1]+mother_rect[3], child_rect[1]+child_rect[3]) - max(mother_rect[1], child_rect[1]))
                        interArea = interW * interH
                        
                        # Calculate parent-child features
                        pm_feature = {
                            'P_IoU': bounding_box_iou(mother_rect, child_rect),
                            'P_M': interArea / area_m if area_m > 0 else 0,
                            'P_D': interArea / area_c if area_c > 0 else 0,
                            'Size_Ratio': area_c / area_m if area_m > 0 else 0
                        }
                        
                        # Check if features meet parent-child criteria
                        ok = True
                        for idx, k in enumerate(param_names):
                            if pm_switch[idx]:
                                vmin, vmax = pm_limits[k]
                                ok &= (pm_feature[k] > vmin and pm_feature[k] < vmax)
                        if ok:
                            child_candidates.append((child, child_rect))
                    
                    # Check if mother cell matches ground truth
                    found_mother = False
                    for gt in gt_cells:
                        gt_rect = rect_from_yolo(gt, img_width, img_height)
                        iou_gt = bounding_box_iou(mother_rect, gt_rect)
                        if iou_gt > 0.5:
                            found_mother = True
                            break
                    
                    # Test all pairs of daughter cells
                    for i in range(len(child_candidates)):
                        for j in range(i+1, len(child_candidates)):
                            rect1 = child_candidates[i][1]
                            rect2 = child_candidates[j][1]
                            area1 = rect1[2]*rect1[3]
                            area2 = rect2[2]*rect2[3]
                            
                            # Calculate intersection between daughter cells
                            interW = max(0, min(rect1[0]+rect1[2], rect2[0]+rect2[2]) - max(rect1[0], rect2[0]))
                            interH = max(0, min(rect1[1]+rect1[3], rect2[1]+rect2[3]) - max(rect1[1], rect2[1]))
                            interArea = interW * interH
                            
                            # Calculate daughter-daughter features
                            dd_feature = {
                                'P_IoU': bounding_box_iou(rect1, rect2),
                                'P_M': interArea / min(area1, area2) if min(area1, area2) > 0 else 0,
                                'P_D': interArea / max(area1, area2) if max(area1, area2) > 0 else 0,
                                'Size_Ratio': area1 / area2 if area2 > 0 else 0
                            }
                            
                            # Check if features meet daughter-daughter criteria
                            ok_dd = True
                            for idx, k in enumerate(param_names):
                                if dd_switch[idx]:
                                    vmin, vmax = dd_limits[k]
                                    ok_dd &= (dd_feature[k] > vmin and dd_feature[k] < vmax)
                            if not ok_dd:
                                continue
                            
                            # Check if daughter cells match ground truth in next frame
                            found1 = any(bounding_box_iou(rect1, rect_from_yolo(gt, img_width, img_height)) > 0.5 for gt in next_gt_cells)
                            found2 = any(bounding_box_iou(rect2, rect_from_yolo(gt, img_width, img_height)) > 0.5 for gt in next_gt_cells)
                            
                            # Count hits and wrong detections
                            if found_mother and found1 and found2:
                                hit_count += 1
                            elif found_mother:
                                wrong_count += 1
                            break
                    if found_mother:
                        break
            
            # Store results for this parameter combination
            row = {**pm_used, **dd_used, 'hit_count': hit_count, 'wrong_count': wrong_count}
            results.append(row)

    # Convert results to DataFrame and calculate metrics
    results_df = pd.DataFrame(results)
    
    # Count total ground truth divisions
    gt_txts = get_files(gt_dir, '.txt')
    N_gt = 0
    for gt_file in gt_txts:
        gt_path = os.path.join(gt_dir, gt_file)
        gt_cells = load_yolo_txt(gt_path)
        if isinstance(gt_cells, np.ndarray):
            N_gt += len(gt_cells)
    if N_gt == 0:
        N_gt = 19  # backup value

    def calc_metrics(row):
        """
        Calculate precision, recall, and F1 score for each parameter combination
        """
        hit = row['hit_count']
        wrong = row['wrong_count']
        pre = hit / (hit + wrong) if (hit + wrong) > 0 else 0
        rec = hit / N_gt if N_gt > 0 else 0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
        return pd.Series({'precision': pre, 'recall': rec, 'F1': f1})
    
    # Calculate metrics and sort by performance
    metrics_df = results_df.apply(calc_metrics, axis=1)
    results_df = pd.concat([results_df, metrics_df], axis=1)
    results_df = results_df.sort_values(['F1', 'hit_count', 'wrong_count'], ascending=[False, False, True])

    # Save only the best parameters
    best_row = results_df.iloc[0]
    param_config = {
        "pm_switch": [int(best_row[f'pm_{k}']) for k in param_names],
        "dd_switch": [int(best_row[f'dd_{k}']) for k in param_names],
        "pm_limits": pm_limits,
        "dd_limits": dd_limits,
        "img_width": img_width,
        "img_height": img_height
    }
    with open(os.path.join(out_dir, "best_param.json"), "w") as f:
        json.dump(param_config, f, indent=2)
    print(f"Best parameters saved to {os.path.join(out_dir, 'best_param.json')}")

if __name__ == '__main__':
    main()
