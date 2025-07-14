import os
import cv2
import numpy as np
import shutil

def yolo_to_rect(yolo, img_w, img_h):
    """
    Convert YOLO format (normalized) to rectangle coordinates (pixel)
    """
    xc, yc, w, h = yolo[1], yolo[2], yolo[3], yolo[4]
    x = (xc - w/2) * img_w
    y = (yc - h/2) * img_h
    w = w * img_w
    h = h * img_h
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))

def rect_to_yolo(rect, img_w, img_h):
    """
    Convert rectangle coordinates (pixel) to YOLO format (normalized)
    """
    x, y, w, h = rect
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return [xc, yc, w_n, h_n]

def convert_16bit_to_8bit(img16):
    """
    Convert 16-bit image to 8-bit and ensure 3-channel BGR format
    """
    img8 = ((img16 - img16.min()) / (img16.max() - img16.min()) * 255).astype(np.uint8)
    if len(img8.shape) == 2:
        img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8

def get_csrt_tracker():
    """
    Create CSRT tracker, handling different OpenCV versions
    """
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ImportError("Your OpenCV doesn't have CSRT tracker, please install: pip install opencv-contrib-python")

def csrt_track(cur_img_path, yolo_label, next_img_path):
    """
    Track object from current frame to next frame using CSRT tracker
    Returns updated YOLO label in next frame or None if tracking fails
    """
    img_cur_16 = cv2.imread(cur_img_path, -1)
    img_next_16 = cv2.imread(next_img_path, -1)
    if img_cur_16 is None or img_next_16 is None:
        print('Image reading failed:', cur_img_path, next_img_path)
        return None
    
    # Convert 16-bit images to 8-bit for tracking
    img_cur = convert_16bit_to_8bit(img_cur_16)
    img_next = convert_16bit_to_8bit(img_next_16)
    img_h, img_w = img_cur.shape[:2]
    
    # Convert YOLO format to rectangle for tracker initialization
    rect = yolo_to_rect(yolo_label, img_w, img_h)
    tracker = get_csrt_tracker()
    tracker.init(img_cur, rect)
    
    # Track to next frame
    success, bbox = tracker.update(img_next)
    if not success:
        return None
    
    # Convert tracked bbox back to YOLO format
    bbox = [int(round(x)) for x in bbox]
    xc, yc, w_n, h_n = rect_to_yolo(bbox, img_w, img_h)
    # Keep class_idx unchanged
    class_idx = int(yolo_label[0])
    return [class_idx, xc, yc, w_n, h_n]

def yolo_iou(label1, label2, img_w, img_h):
    """
    Calculate IoU between two YOLO format labels
    """
    r1 = yolo_to_rect(label1, img_w, img_h)
    r2 = yolo_to_rect(label2, img_w, img_h)
    xA = max(r1[0], r2[0])
    yA = max(r1[1], r2[1])
    xB = min(r1[0]+r1[2], r2[0]+r2[2])
    yB = min(r1[1]+r1[3], r2[1]+r2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    if interArea == 0:
        return 0
    iou = interArea / (area1 + area2 - interArea)
    return iou

def bf2gfp(bf):
    """
    Convert Bright Field filename to corresponding GFP filename
    """
    return bf.replace('Bright Field', 'GFP').replace('.txt', '.tif').replace('03_3', '03_2')

def get_frame_idx(name):
    """
    Extract frame index from filename
    """
    return int(os.path.splitext(name)[0].split('_')[-1])

def set_frame_idx(name, idx):
    """
    Set frame index in filename
    """
    prefix = '_'.join(os.path.splitext(name)[0].split('_')[:-1])
    ext = os.path.splitext(name)[1]
    return f"{prefix}_{idx:03d}{ext}"

def all_gfp_names(img_dir):
    """
    Get all GFP image filenames in directory
    """
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.tif')])

def read_yolo_txt(txt_path):
    """
    Read YOLO format labels from text file
    """
    if not os.path.exists(txt_path) or txt_path.endswith('classes.txt'):
        return []
    arr = np.loadtxt(txt_path)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def save_yolo_txt(txt_path, labels):
    """
    Save YOLO format labels to text file
    """
    with open(txt_path, 'w') as f:
        for label in labels:
            f.write('%d %.6f %.6f %.6f %.6f\n' % tuple(label))

def draw_boxes_on_image(img_path, labels, class_colors, out_path):
    """
    Draw bounding boxes on image and save visualization
    """
    img16 = cv2.imread(img_path, -1)
    if img16 is None:
        return
    img = convert_16bit_to_8bit(img16)
    img_h, img_w = img.shape[:2]
    for i, label in enumerate(labels):
        rect = yolo_to_rect(label, img_w, img_h)
        color = class_colors.get(int(label[0]), (0,255,0))
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 2)
        cv2.putText(img, str(int(label[0])), (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imwrite(out_path, img)

def main():
    """
    Main function for cell division trajectory CSRT tracking with FL label IoU synchronization
    """
    import argparse
    parser = argparse.ArgumentParser(description="Cell division trajectory CSRT tracking with FL label IoU synchronization")
    parser.add_argument('--node_dir', type=str, required=True, help='Division node label directory (BF names)')
    parser.add_argument('--gfp_dir', type=str, required=True, help='GFP 16-bit image directory')
    parser.add_argument('--fl_label_dir', type=str, required=True, help='FL label txt directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output folder')
    parser.add_argument('--class_file', type=str, required=True, help='classes.txt path')
    parser.add_argument('--min_frame', type=int, default=1, help='Minimum frame number')
    parser.add_argument('--max_frame', type=int, default=None, help='Maximum frame number')
    parser.add_argument('--save_vis', action='store_true', help='Whether to save visualization')
    args = parser.parse_args()

    node_dir = args.node_dir
    gfp_dir = args.gfp_dir
    fl_label_dir = args.fl_label_dir
    out_dir = args.out_dir
    class_file = args.class_file
    min_frame = args.min_frame
    max_frame = args.max_frame
    save_vis = args.save_vis

    # Create output directory and copy classes.txt
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(class_file, os.path.join(out_dir, 'classes.txt'))

    # Get all node label files and GFP images
    node_txts = sorted([f for f in os.listdir(node_dir) if f.endswith('.txt') and f != 'classes.txt'])
    all_gfp = all_gfp_names(gfp_dir)
    
    # Determine maximum frame number if not specified
    if max_frame is None:
        if all_gfp:
            max_frame = max([get_frame_idx(n) for n in all_gfp])
        else:
            print('GFP folder is empty!')
            return

    # Process each node label file
    for txt_name in node_txts:
        labels = read_yolo_txt(os.path.join(node_dir, txt_name))
        if len(labels) == 0:
            continue

        # Find mother cells (class ID divisible by 3)
        mothers = [lab for lab in labels if int(lab[0]) % 3 == 0]
        if len(mothers) == 0:
            print(f"{txt_name}: No mother cell labels found, skipping.")
            continue

        # Process each mother cell
        for mother in mothers:
            m_class = int(mother[0])
            folder = os.path.join(out_dir, f"cell_{txt_name.replace('.txt','')}_m{m_class}")
            os.makedirs(folder, exist_ok=True)
            tracks = {}

            # === 1. Track mother cell backward ===
            frame = get_frame_idx(txt_name)
            cur_gfp = bf2gfp(txt_name)
            cur_path = os.path.join(gfp_dir, cur_gfp)
            last_label = mother.copy()
            img_h, img_w = None, None
            
            # Track mother cell from current frame backward to min_frame
            while frame >= min_frame:
                if frame not in tracks:
                    tracks[frame] = [None, None, None]
                tracks[frame][0] = last_label.copy()
                print(f'[TRACK-MOTHER] class={m_class}, frame={frame:03d}')
                
                next_frame = frame - 1
                if next_frame < min_frame:
                    break
                    
                next_gfp = set_frame_idx(cur_gfp, next_frame)
                next_path = os.path.join(gfp_dir, next_gfp)
                pred_label = csrt_track(cur_path, last_label, next_path)
                
                if pred_label is None:
                    print('  -> CSRT tracking failed, mother trajectory terminated')
                    break
                    
                print(f'[Tracking Prediction] Current frame={next_gfp}, CSRT prediction: [{pred_label[1]:.4f}, {pred_label[2]:.4f}, {pred_label[3]:.4f}, {pred_label[4]:.4f}]')
                
                # Try to match with FL labels using IoU
                fl_label_file = os.path.join(fl_label_dir, next_gfp.replace('.tif','.txt'))
                print(f'[FL Labels] Reading FL label file: {fl_label_file}')
                fl_labels = read_yolo_txt(fl_label_file)
                best_label = None
                best_iou = 0
                
                if fl_labels is not None and len(fl_labels) > 0:
                    if img_h is None or img_w is None:
                        img16 = cv2.imread(next_path, -1)
                        img_h, img_w = img16.shape[:2]
                    for idx, fl in enumerate(fl_labels):
                        iou = yolo_iou(pred_label, fl, img_w, img_h)
                        print(f'    FL label#{idx}: [{fl[1]:.4f},{fl[2]:.4f},{fl[3]:.4f},{fl[4]:.4f}], IoU={iou:.4f}')
                        if iou > 0.5 and iou > best_iou:
                            best_iou = iou
                            best_label = fl
                            
                if best_label is not None:
                    print(f'  -> IoU matched FL label, IoU={best_iou:.4f}, using FL label position but keeping class unchanged')
                    last_label[1:] = best_label[1:]  # Only override position
                else:
                    print('  -> No FL label with IoU>0.5, using CSRT prediction box directly')
                    last_label = pred_label.copy()
                    
                frame = next_frame
                cur_path = next_path

            # === 2. Track two daughter cells forward ===
            start_frame = get_frame_idx(txt_name) + 1
            start_txt = set_frame_idx(txt_name, start_frame)
            next_labels = read_yolo_txt(os.path.join(node_dir, start_txt))
            daughters = []
            
            # Find daughter cells with corresponding class IDs
            if len(next_labels) > 0:
                for d in next_labels:
                    if int(d[0]) == m_class + 1:
                        daughters.append((1, d.copy()))
                    elif int(d[0]) == m_class + 2:
                        daughters.append((2, d.copy()))
                        
            if len(daughters) < 2:
                print(f"{txt_name}: Could not find two daughter cells, mother cell {m_class} only tracks mother trajectory.")

            # Track each daughter cell forward
            for d_idx, daughter in daughters:
                frame = start_frame
                cur_gfp = bf2gfp(start_txt)
                cur_path = os.path.join(gfp_dir, cur_gfp)
                last_label = daughter.copy()
                daughter_class = int(daughter[0])
                img_h, img_w = None, None
                
                # Track daughter cell from start_frame forward to max_frame
                while frame <= max_frame:
                    if frame not in tracks:
                        tracks[frame] = [None, None, None]
                    tracks[frame][d_idx] = last_label.copy()
                    print(f'[TRACK-DAUGHTER] class={daughter_class}, frame={frame:03d}')
                    
                    next_frame = frame + 1
                    if next_frame > max_frame:
                        break
                        
                    next_gfp = set_frame_idx(cur_gfp, next_frame)
                    next_path = os.path.join(gfp_dir, next_gfp)
                    pred_label = csrt_track(cur_path, last_label, next_path)
                    
                    if pred_label is None:
                        print(f'  -> CSRT tracking failed, daughter trajectory terminated')
                        break
                        
                    print(f'[Tracking Prediction] Current frame={next_gfp}, CSRT prediction: [{pred_label[1]:.4f}, {pred_label[2]:.4f}, {pred_label[3]:.4f}, {pred_label[4]:.4f}]')
                    
                    # Try to match with FL labels using IoU
                    fl_label_file = os.path.join(fl_label_dir, next_gfp.replace('.tif','.txt'))
                    print(f'[FL Labels] Reading FL label file: {fl_label_file}')
                    fl_labels = read_yolo_txt(fl_label_file)
                    best_label = None
                    best_iou = 0
                    
                    if fl_labels is not None and len(fl_labels) > 0:
                        if img_h is None or img_w is None:
                            img16 = cv2.imread(next_path, -1)
                            img_h, img_w = img16.shape[:2]
                        for idx, fl in enumerate(fl_labels):
                            iou = yolo_iou(pred_label, fl, img_w, img_h)
                            print(f'    FL label#{idx}: [{fl[1]:.4f},{fl[2]:.4f},{fl[3]:.4f},{fl[4]:.4f}], IoU={iou:.4f}')
                            if iou > 0.5 and iou > best_iou:
                                best_iou = iou
                                best_label = fl
                                
                    if best_label is not None:
                        print(f'  -> IoU matched FL label, IoU={best_iou:.4f}, using FL label position but keeping class unchanged')
                        # ---------Key: only change position info, not class---------
                        last_label[1:] = best_label[1:]
                    else:
                        print(f'  -> No FL label with IoU>0.5, using CSRT prediction box directly')
                        last_label = pred_label.copy()
                    # ---------Always keep last_label[0] unchanged!----------
                    frame = next_frame
                    cur_path = next_path

            # === 4. Output all trajectory targets for each frame ===
            class_colors = {m_class: (0,0,255), m_class+1: (255,0,0), m_class+2: (0,255,0)}
            for frame_id, labs in tracks.items():
                out_txt = set_frame_idx(bf2gfp(txt_name).replace('.tif','.txt'), frame_id)
                labs_to_save = [lab for lab in labs if lab is not None]
                if labs_to_save:
                    save_yolo_txt(os.path.join(folder, out_txt), labs_to_save)
                    if save_vis:
                        vis_img = os.path.join(gfp_dir, set_frame_idx(bf2gfp(txt_name), frame_id))
                        vis_out = os.path.join(folder, out_txt.replace('.txt', '_vis.png'))
                        draw_boxes_on_image(vis_img, labs_to_save, class_colors, vis_out)

    print("All cell trajectory tracking completed, results saved in", out_dir)

if __name__ == '__main__':
    main()
