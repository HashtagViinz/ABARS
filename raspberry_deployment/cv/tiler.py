import os
import cv2
import yaml
from pathlib import Path
from logger import log

def build_tiled_dataset(source_yaml: str, overlap_ratio: float = 0.1) -> str:
    """
    Legge il dataset originale specificato nel file yaml, 
    effettua uno slicing 2x2 delle immagini con un margine di overlap.
    Inoltre calcola le frequenze delle classi e fa un oversampling (massimo 5x)
    delle piastrelle di TRAIN che contengono classi rare (es. piscine).
    """
    source_path = Path(source_yaml)
    if not source_path.exists():
        log(f"File yaml sorgente non trovato: {source_yaml}", "red")
        return source_yaml
        
    with open(source_path, "r") as f:
        data_yaml = yaml.safe_load(f)
        
    dataset_base = source_path.parent
    new_dir_name = dataset_base.name + "_tiled"
    target_dir = Path(new_dir_name)
    new_yaml_path = target_dir / f"{source_path.stem}_tiled.yaml"
    
    if target_dir.exists() and new_yaml_path.exists():
        log(f"Il dataset {target_dir} esiste gia'. Nessuna generazione necessaria.", "yellow")
        return str(new_yaml_path)
        
    log(f"Inizio generazione del dataset piastrellato: {target_dir}", "blue")
    
    # --- STEP 1: Conteggio classi globale per Oversampling ---
    class_counts = {}
    train_lbl_dir = dataset_base / "labels" / "train"
    if train_lbl_dir.exists():
        for lbl_file in train_lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        
    log(f"Frequenze classi nel TRAIN: {class_counts}", "cyan")
    
    multipliers = {}
    if class_counts:
        max_count = max(class_counts.values())
        for cls_id, count in class_counts.items():
            mult = max_count / count if count > 0 else 1
            # Limitiamo il duplicatore a massimo 5 copie per non far esplodere lo storage
            multipliers[cls_id] = min(max(1, int(mult)), 5)
    log(f"Moltiplicatori Oversampling applicati: {multipliers}", "yellow")
    # ---------------------------------------------------------
    
    for split in ["train", "val", "test"]:
        split_img_dir_rel = data_yaml.get(split, f"images/{split}")
        src_img_dir = dataset_base / split_img_dir_rel
        src_lbl_dir = dataset_base / "labels" / split
        
        target_img_dir = target_dir / "images" / split
        target_lbl_dir = target_dir / "labels" / split
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
            
        images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
        total_imgs = len(images)
        if total_imgs == 0:
            continue
        
        log(f"Processando {total_imgs} immagini per lo split '{split}'...", "cyan")
        
        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None: continue
                
            img_h, img_w = img.shape[:2]
            
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            boxes = []
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_min = (float(parts[1]) - float(parts[3]) / 2) * img_w
                            x_max = (float(parts[1]) + float(parts[3]) / 2) * img_w
                            y_min = (float(parts[2]) - float(parts[4]) / 2) * img_h
                            y_max = (float(parts[2]) + float(parts[4]) / 2) * img_h
                            boxes.append((cls_id, x_min, y_min, x_max, y_max))
                            
            overlap_x = int(img_w * overlap_ratio)
            overlap_y = int(img_h * overlap_ratio)
            tiles = [
                (0, 0, img_w // 2 + overlap_x, img_h // 2 + overlap_y),
                (img_w // 2 - overlap_x, 0, img_w, img_h // 2 + overlap_y),
                (0, img_h // 2 - overlap_y, img_w // 2 + overlap_x, img_h),
                (img_w // 2 - overlap_x, img_h // 2 - overlap_y, img_w, img_h)
            ]
            
            for t_idx, (tx_min, ty_min, tx_max, ty_max) in enumerate(tiles):
                tx_min = max(0, tx_min)
                ty_min = max(0, ty_min)
                tx_max = min(img_w, tx_max)
                ty_max = min(img_h, ty_max)
                
                cropped_w = tx_max - tx_min
                cropped_h = ty_max - ty_min
                if cropped_w <= 0 or cropped_h <= 0: continue
                    
                new_boxes = []
                tile_classes = set()
                
                for (cls_id, bx_min, by_min, bx_max, by_max) in boxes:
                    inter_x_min = max(bx_min, tx_min)
                    inter_y_min = max(by_min, ty_min)
                    inter_x_max = min(bx_max, tx_max)
                    inter_y_max = min(by_max, ty_max)
                    
                    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                        rel_x_min, rel_y_min = inter_x_min - tx_min, inter_y_min - ty_min
                        rel_x_max, rel_y_max = inter_x_max - tx_min, inter_y_max - ty_min
                        
                        new_x_c = ((rel_x_min + rel_x_max) / 2) / cropped_w
                        new_y_c = ((rel_y_min + rel_y_max) / 2) / cropped_h
                        new_w = (rel_x_max - rel_x_min) / cropped_w
                        new_h = (rel_y_max - rel_y_min) / cropped_h
                        
                        new_boxes.append((cls_id, new_x_c, new_y_c, new_w, new_h))
                        tile_classes.add(cls_id)
                        
                cropped_img = img[ty_min:ty_max, tx_min:tx_max]
                
                # Calcola quante copie salvare (solo in TRAIN)
                copies = 1
                if split == "train" and tile_classes:
                    max_tile_mult = max(multipliers.get(c, 1) for c in tile_classes)
                    copies = max_tile_mult
                
                for c_idx in range(copies):
                    tile_name = f"{img_path.stem}_t{t_idx}_c{c_idx}" if copies > 1 else f"{img_path.stem}_t{t_idx}"
                    tile_img_path = target_img_dir / f"{tile_name}.jpg"
                    tile_lbl_path = target_lbl_dir / f"{tile_name}.txt"
                    
                    cv2.imwrite(str(tile_img_path), cropped_img)
                    with open(tile_lbl_path, "w") as f:
                        for box in new_boxes:
                            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                            
            if (idx + 1) % 100 == 0 or (idx + 1) == total_imgs:
                log(f"[{split}] Progresso: {idx + 1}/{total_imgs} immagini affettate", "blue")
                
    # --- STEP 2: Ricalcolo delle frequenze finali (Post-Oversampling) ---
    new_class_counts = {}
    new_train_lbl_dir = target_dir / "labels" / "train"
    if new_train_lbl_dir.exists():
        for lbl_file in new_train_lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        new_class_counts[cls_id] = new_class_counts.get(cls_id, 0) + 1
                        
    if new_class_counts:
        total_new = sum(new_class_counts.values())
        log(f"--- DISTRIBUZIONE CLASSI FINALE (TRAIN) ---", "green")
        for cls_id, count in sorted(new_class_counts.items()):
            pct = (count / total_new) * 100
            log(f"Classe {cls_id}: {count} istanze ({pct:.2f}%)", "green")
        log(f"-------------------------------------------", "green")
    # --------------------------------------------------------------------

    data_yaml['path'] = str(target_dir.absolute())
    with open(new_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    log(f"Generazione Tiling completata! Dataset pronto in {target_dir}", "green")
    return str(new_yaml_path)

if __name__ == "__main__":
    build_tiled_dataset("dataset/uavod10.yaml")
