import os
import cv2
import yaml
from pathlib import Path
from logger import log

def build_tiled_dataset(source_yaml: str, overlap_ratio: float = 0.1) -> str:
    """
    Legge il dataset originale specificato nel file yaml, 
    effettua uno slicing 2x2 delle immagini con un margine di overlap,
    ricalcola le coordinate dei bounding box per ogni nuova piastrella,
    scarta le piastrelle vuote (senza oggetti) e salva il nuovo dataset.
    """
    source_path = Path(source_yaml)
    if not source_path.exists():
        log(f"File yaml sorgente non trovato: {source_yaml}", "red")
        return source_yaml
        
    with open(source_path, "r") as f:
        data_yaml = yaml.safe_load(f)
        
    dataset_base = source_path.parent
    
    # Determiniamo il nome della nuova cartella (nella root del progetto)
    new_dir_name = dataset_base.name + "_tiled"
    target_dir = Path(new_dir_name)
    new_yaml_path = target_dir / f"{source_path.stem}_tiled.yaml"
    
    if target_dir.exists() and new_yaml_path.exists():
        log(f"Il dataset {target_dir} esiste gia'. Nessuna generazione necessaria.", "yellow")
        return str(new_yaml_path)
        
    log(f"Inizio generazione del dataset piastrellato: {target_dir}", "blue")
    
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
            raise FileNotFoundError(f"Nessuna immagine trovata in {src_img_dir}. Assicurati che i path in {source_yaml} siano corretti.")
        
        log(f"Processando {total_imgs} immagini per lo split '{split}'...", "cyan")
        
        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            
            # Leggi etichette
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            boxes = []
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_c = float(parts[1])
                            y_c = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # Denormalizza
                            x_min = (x_c - w / 2) * img_w
                            x_max = (x_c + w / 2) * img_w
                            y_min = (y_c - h / 2) * img_h
                            y_max = (y_c + h / 2) * img_h
                            boxes.append((cls_id, x_min, y_min, x_max, y_max))
                            
            # Calcola le coordinate delle piastrelle (2x2 grid)
            overlap_x = int(img_w * overlap_ratio)
            overlap_y = int(img_h * overlap_ratio)
            
            tiles = [
                (0, 0, img_w // 2 + overlap_x, img_h // 2 + overlap_y), # Top-Left
                (img_w // 2 - overlap_x, 0, img_w, img_h // 2 + overlap_y), # Top-Right
                (0, img_h // 2 - overlap_y, img_w // 2 + overlap_x, img_h), # Bottom-Left
                (img_w // 2 - overlap_x, img_h // 2 - overlap_y, img_w, img_h) # Bottom-Right
            ]
            
            for t_idx, (tx_min, ty_min, tx_max, ty_max) in enumerate(tiles):
                # Ensure bounds
                tx_min = max(0, tx_min)
                ty_min = max(0, ty_min)
                tx_max = min(img_w, tx_max)
                ty_max = min(img_h, ty_max)
                
                cropped_w = tx_max - tx_min
                cropped_h = ty_max - ty_min
                
                if cropped_w <= 0 or cropped_h <= 0:
                    continue
                    
                new_boxes = []
                for (cls_id, bx_min, by_min, bx_max, by_max) in boxes:
                    # Intersezione col tile
                    inter_x_min = max(bx_min, tx_min)
                    inter_y_min = max(by_min, ty_min)
                    inter_x_max = min(bx_max, tx_max)
                    inter_y_max = min(by_max, ty_max)
                    
                    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                        # Converti in coordinate relative alla piastrella
                        rel_x_min = inter_x_min - tx_min
                        rel_y_min = inter_y_min - ty_min
                        rel_x_max = inter_x_max - tx_min
                        rel_y_max = inter_y_max - ty_min
                        
                        # Ri-normalizza
                        new_x_c = ((rel_x_min + rel_x_max) / 2) / cropped_w
                        new_y_c = ((rel_y_min + rel_y_max) / 2) / cropped_h
                        new_w = (rel_x_max - rel_x_min) / cropped_w
                        new_h = (rel_y_max - rel_y_min) / cropped_h
                        
                        new_boxes.append((cls_id, new_x_c, new_y_c, new_w, new_h))
                        
                # Salviamo SEMPRE il tile (Negative Samples per YOLO)
                tile_name = f"{img_path.stem}_t{t_idx}"
                tile_img_path = target_img_dir / f"{tile_name}.jpg"
                tile_lbl_path = target_lbl_dir / f"{tile_name}.txt"
                
                cropped_img = img[ty_min:ty_max, tx_min:tx_max]
                cv2.imwrite(str(tile_img_path), cropped_img)
                
                with open(tile_lbl_path, "w") as f:
                    for box in new_boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                            
            if (idx + 1) % 100 == 0 or (idx + 1) == total_imgs:
                log(f"[{split}] Progresso: {idx + 1}/{total_imgs} immagini affettate", "blue")
                
    # Creiamo il nuovo file yaml
    data_yaml['path'] = str(target_dir.absolute())
    with open(new_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    log(f"Generazione Tiling completata! Dataset pronto in {target_dir}", "green")
    return str(new_yaml_path)

if __name__ == "__main__":
    build_tiled_dataset("dataset/uavod10.yaml")
