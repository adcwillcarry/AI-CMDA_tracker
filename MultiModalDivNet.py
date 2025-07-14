import os
import numpy as np
import argparse
import json

def main():
    """
    Main function to find division nodes based on parameters and output YOLO labels
    """
    parser = argparse.ArgumentParser(description="Find division nodes based on parameters and output YOLO labels")
    parser.add_argument('--bf_dir', type=str, required=True, help="Bright Field label folder path")
    parser.add_argument('--fl_dir', type=str, required=True, help="Fluorescence label folder path")
    parser.add_argument('--param_json', type=str, default='output/best_param.json', help="Ablation parameters and range JSON file")
    parser.add_argument('--output_dir', type=str, default='output/nodes', help="Output folder path")
    args = parser.parse_args()
    bf_dir, fl_dir = args.bf_dir, args.fl_dir
    param_json = args.param_json
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load configuration parameters from JSON file
    with open(param_json) as f:
        config = json.load(f)
    pm_switch, dd_switch = config['pm_switch'], config['dd_switch']
    pm_limits, dd_limits = config['pm_limits'], config['dd_limits']
    img_width = config.get('img_width', 1224)
    img_height = config.get('img_height', 904)
    param_names = ['P_IoU', 'P_M', 'P_D', 'Size_Ratio']

    def get_fl_name(bf_name):
        """
        Convert bright field filename to corresponding fluorescence filename
        """
        fl_name = bf_name.replace('_3_', '_2_').replace('Bright Field', 'GFP')
        prefix, num_txt = fl_name.rsplit('_', 1)
        num, ext = os.path.splitext(num_txt)
        num_new = f"{int(num)+1:03d}{ext}"
        return f"{prefix}_{num_new}"

    def get_next_bf_name(bf_name):
        """
        Get the next frame's bright field filename
        """
        prefix, num_txt = bf_name.rsplit('_', 1)
        num, ext = os.path.splitext(num_txt)
        num_new = f"{int(num)+1:03d}{ext}"
        return f"{prefix}_{num_new}"

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

    def rect_from_yolo(label):
        """
        Convert YOLO format (normalized) to rectangle coordinates (pixel)
        Returns [x_min, y_min, width, height]
        """
        x_center = label[1] * img_width
        y_center = label[2] * img_height
        width = label[3] * img_width
        height = label[4] * img_height
        x_min = round(x_center - width / 2)
        y_min = round(y_center - height / 2)
        return [x_min, y_min, width, height]

    def bounding_box_iou(rect1, rect2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
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

    # Get all bright field text files (excluding classes.txt)
    bf_txts = sorted([f for f in os.listdir(bf_dir) if f.endswith('.txt') and f != "classes.txt"])

    # Store the real class_id order (mother cell/daughter cell) for each category
    # Finally write classes.txt uniformly
    class_id_list = []
    class_triplets = []  # [(mother_id, daughter1_id, daughter2_id), ...]

    # First, count all division nodes (order is important, ensure sequence 0,1,2,3.. is available in python)
    triplet_list = []
    # Process each bright field image to find division events
    for idx, bf_file in enumerate(bf_txts):
        bf_path = os.path.join(bf_dir, bf_file)
        fl_file = get_fl_name(bf_file)
        fl_path = os.path.join(fl_dir, fl_file)
        if not os.path.exists(fl_path):
            continue
        
        # Load mother cells from bright field and child cells from fluorescence
        mother_cells = load_yolo_txt(bf_path)
        child_cells = load_yolo_txt(fl_path)
        found_this_img = False
        
        # For each mother cell, find potential daughter cells
        for mother in mother_cells:
            mother_rect = rect_from_yolo(mother)
            child_candidates = []
            
            # Check each child cell against parent-child criteria
            for child in child_cells:
                child_rect = rect_from_yolo(child)
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
                for idx_p, k in enumerate(param_names):
                    if pm_switch[idx_p]:
                        vmin, vmax = pm_limits[k]
                        ok &= (pm_feature[k] > vmin and pm_feature[k] < vmax)
                if ok:
                    child_candidates.append((child, child_rect))
            
            # Find pairs of daughter cells
            for i in range(len(child_candidates)):
                for j in range(i+1, len(child_candidates)):
                    rect1 = child_candidates[i][1]
                    rect2 = child_candidates[j][1]
                    area1 = rect1[2]*rect1[3]
                    area2 = rect2[2]*rect2[3]
                    
                    # Calculate intersection between two daughter cells
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
                    for idx_d, k in enumerate(param_names):
                        if dd_switch[idx_d]:
                            vmin, vmax = dd_limits[k]
                            ok_dd &= (dd_feature[k] > vmin and dd_feature[k] < vmax)
                    if not ok_dd:
                        continue
                    
                    # Generate unique IDs for mother and daughter cells
                    mother_id = 2 + len(triplet_list) * 10
                    daughter1_id = mother_id * 10
                    daughter2_id = mother_id * 10 + 1
                    class_triplets.append((mother_id, daughter1_id, daughter2_id))
                    triplet_list.append({
                        "bf_file": bf_file,
                        "next_bf": get_next_bf_name(bf_file),
                        "mother": mother,
                        "daughter1": child_candidates[i][0],
                        "daughter2": child_candidates[j][0]
                    })
                    found_this_img = True
                    break
            if found_this_img:
                break

    # Generate classes.txt mapping (mother cell - daughter cell, order is index)
    all_classes = []
    for trio in class_triplets:
        all_classes.extend(trio)
    # Write classes.txt file
    with open(os.path.join(out_dir, "classes.txt"), 'w') as f:
        for cid in all_classes:
            f.write(f"{cid}\n")

    # Create mapping from class ID to classes.txt line number (index)
    classid2idx = {cid: idx for idx, cid in enumerate(all_classes)}

    # Save labels for each image (mother cells, daughter cells)
    # First aggregate output: {txt filename: [ [class_idx, x_center, y_center, w, h], ... ] }
    img_label_map = {}

    # Process each division triplet to generate labels
    for i, triplet in enumerate(triplet_list):
        mother_id, daughter1_id, daughter2_id = class_triplets[i]
        
        # Mother cell: current frame (bf_file)
        bf_file = triplet['bf_file']
        m_raw = triplet['mother']
        m_idx = classid2idx[mother_id]
        m_label = [m_idx, m_raw[1], m_raw[2], m_raw[3], m_raw[4]]
        if bf_file not in img_label_map:
            img_label_map[bf_file] = []
        img_label_map[bf_file].append(m_label)
        
        # Daughter cells: next frame (next_bf)
        next_bf_file = triplet['next_bf']
        d1_raw = triplet['daughter1']
        d2_raw = triplet['daughter2']
        d1_idx = classid2idx[daughter1_id]
        d2_idx = classid2idx[daughter2_id]
        d1_label = [d1_idx, d1_raw[1], d1_raw[2], d1_raw[3], d1_raw[4]]
        d2_label = [d2_idx, d2_raw[1], d2_raw[2], d2_raw[3], d2_raw[4]]
        if next_bf_file not in img_label_map:
            img_label_map[next_bf_file] = []
        img_label_map[next_bf_file].append(d1_label)
        img_label_map[next_bf_file].append(d2_label)

    # Write all label txt files in YOLO format: class_idx cx cy w h
    for txt_file, label_list in img_label_map.items():
        out_path = os.path.join(out_dir, txt_file)
        with open(out_path, "w") as f:
            for lab in label_list:
                f.write("%d %.6f %.6f %.6f %.6f\n" % tuple(lab))
    print(f"YOLO labels and classes.txt have been saved to {out_dir}")

if __name__ == '__main__':
    main()
