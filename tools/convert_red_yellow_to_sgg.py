import os
import json
import glob
from PIL import Image

def yolobbox2bbox(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    w1 = w * img_w
    h1 = h * img_h
    return [x1, y1, w1, h1]

def process_split(split="train"):
    base_dir = r"e:\BAS\realtime-work-detection\dataset\red_yellow_dataset"
    img_dir = os.path.join(base_dir, "images", split)
    lbl_dir = os.path.join(base_dir, "labels", split)
    
    classes = {0: "container_box", 1: "container_lid", 2: "red_box", 3: "yellow_box", 4: "operator_hand", 5: "human_body"}
    
    images = []
    annotations = []
    rel_annotations = []
    
    predicates = ["inside", "outside", "holding"]
    pred_ids = {p: i for i, p in enumerate(predicates)}
    
    ann_id = 0
    img_id = 0
    
    for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path):
            continue
            
        try:
            im = Image.open(img_path)
            W, H = im.size
        except:
            continue
            
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})
        
        # Read boxes
        local_boxes = {}
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    bbox = yolobbox2bbox(x, y, w, h, W, H)
                    
                    local_boxes[cls_id] = {"id": ann_id, "bbox": bbox, "name": classes[cls_id]}
                    
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "bbox": bbox,
                        "category_id": cls_id + 1
                    })
                    ann_id += 1
        
        # Synthesize relationships
        cont = local_boxes.get(0)
        red = local_boxes.get(2)
        yel = local_boxes.get(3)
        hand = local_boxes.get(4)
        
        if cont and red:
            # check inside/outside
            cx, cy, cw, ch = cont["bbox"]
            rx, ry, rw, rh = red["bbox"]
            rcx, rcy = rx + rw/2, ry + rh/2
            
            is_out = rcx < cx - 20 or rcx > cx + cw + 20 or rcy < cy - 20
            rel_annotations.append({
                "image_id": img_id,
                "subject_id": red["id"],
                "object_id": cont["id"],
                "predicate_id": pred_ids["outside" if is_out else "inside"]
            })
            
        if cont and yel:
            cx, cy, cw, ch = cont["bbox"]
            yx, yy, yw, yh = yel["bbox"]
            ycx, ycy = yx + yw/2, yy + yh/2
            
            is_out = ycx < cx - 20 or ycx > cx + cw + 20 or ycy < cy - 20
            rel_annotations.append({
                "image_id": img_id,
                "subject_id": yel["id"],
                "object_id": cont["id"],
                "predicate_id": pred_ids["outside" if is_out else "inside"]
            })
            
        if hand and red:
            # check holding
            hx, hy, hw, hh = hand["bbox"]
            rx, ry, rw, rh = red["bbox"]
            
            # Simple overlap check
            if not (rx > hx+hw or rx+rw < hx or ry > hy+hh or ry+rh < hy):
                rel_annotations.append({
                    "image_id": img_id,
                    "subject_id": hand["id"],
                    "object_id": red["id"],
                    "predicate_id": pred_ids["holding"]
                })
                
        if hand and yel:
            hx, hy, hw, hh = hand["bbox"]
            yx, yy, yw, yh = yel["bbox"]
            if not (yx > hx+hw or yx+yw < hx or yy > hy+hh or yy+yh < hy):
                rel_annotations.append({
                    "image_id": img_id,
                    "subject_id": hand["id"],
                    "object_id": yel["id"],
                    "predicate_id": pred_ids["holding"]
                })
                
        img_id += 1
        
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i+1, "name": n} for i, n in classes.items()],
        "rel_categories": [{"id": i, "name": n} for n, i in pred_ids.items()],
        "rel_annotations": rel_annotations
    }
    
    out_path = os.path.join(base_dir, f"sgg_{split}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[{split}] Generated {len(images)} images, {len(annotations)} boxes, {len(rel_annotations)} relations.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
