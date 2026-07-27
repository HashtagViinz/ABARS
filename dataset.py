# Scirpt to Manage the Dataset for YOLO

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from torchvision.transforms import v2
from logger import log
from pathlib import Path
import random
from collections import Counter
import shutil

# MAP of class IDs to YOLO class indices
CLASS_MAPPING = {
    35694: 0, # building
    35697: 1, # cable-tower
    35700: 2, # cultivation-mesh-cage
    35696: 3, # landslide
    35701: 4, # pool
    35695: 5, # prefabricated-house
    35699: 6, # quarry
    35702: 7, # ship
    35698: 8, # vehicle
    35693: 9  # well
}



class UAVOD(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None) -> None:
        """
        Costruttore per il dataset UAVOD.
        Args:
            images_dir (str): Percorso alla cartella contenente le immagini.
            annotations_dir (str): Percorso alla cartella contenente le annotazioni in formato JSON.
            transforms (callable, optional): Trasformazioni da applicare alle immagini e alle annotazioni.
        """
        
        self.images_dir = images_dir            # Directory delle immagini
        self.annotations_dir = annotations_dir  # Directory delle annotazioni
        self.transforms = transforms            # 
        
        self.image_files = sorted(os.listdir(images_dir))   # Lista dei nomi dei file delle immagini, ordinata alfabeticamente

        # Costruzione mapping classi -> indice
        self.class_map = self._build_class_map()# Scorriamo ogni file di annotazione per costruire un dizionario che mappa i nomi delle classi.

    def _build_class_map(self) -> dict:
        """
        Funzione che costruisce un dizionario che mappa i nomi delle classi a indici interi.
        Ritorna:
            dict: Dizionario con chiavi come nomi delle classi e valori come indici.
        """
        classes = set()
        for file in os.listdir(self.annotations_dir):
            with open(os.path.join(self.annotations_dir, file)) as f:
                data = json.load(f)
                for obj in data["objects"]:
                    classes.add(obj["classTitle"])

        classes = sorted(list(classes))
        return {cls: i+1 for i, cls in enumerate(classes)}  # 0 è background

    def __len__(self)-> int:
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[Any, Dict[str, Any]]:
        """
        Trova il nome dell'immagine corrispondente all'indice richiesto,
        la carica usando la libreria PIL e apre il rispettivo file JSON delle annotazioni.
        
        Args:
            idx (int): Indice dell'immagine da caricare.+
        """
        
        
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # Assicurati che l'estensione sia corretta. Se le img sono .png, ann_path cerca "nome.png.json"
        ann_path = os.path.join(self.annotations_dir, img_name + ".json")

        img = Image.open(img_path).convert("RGB")
        
        # Recupera larghezza e altezza prima di qualsiasi trasformazione
        w, h = img.size 

        with open(ann_path) as f:
            data = json.load(f)

        boxes = []
        labels = []

        for obj in data["objects"]:
            (xmin, ymin), (xmax, ymax) = obj["points"]["exterior"]
            
            # --- SANITIZZAZIONE BOXES ---
            # 1. Evita coordinate invertite (es. xmin maggiore di xmax)
            x1 = min(xmin, xmax)
            x2 = max(xmin, xmax)
            y1 = min(ymin, ymax)
            y2 = max(ymin, ymax)
            
            # 2. Taglia ai bordi se una box "esce" dall'immagine
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 3. Ignora le box degeneri (larghezza o altezza = 0)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_map[obj["classTitle"]])

        # --- FIX: GESTIONE IMMAGINI VUOTE ---
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # --- FIX: APPLICAZIONE TRANSFORMS ---
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target
    
    def analyze_class_distribution() -> None:
        """
        Function that analyze the distribution of classes in the dataset.
        Then it will plot the distribution of classes in a bar chart.
        """
        pass       
    

def process_and_split_dataset(dataset_dir: str, test_ratio: float = 0.1, val_ratio: float = 0.2, seed: int = 42) -> None:
    """ 
    Converts JSON annotations to YOLO format and creates train/val/test folders.
    Keeps the original dataset unchanged and uses copy to reproduce it.   
    The function will create two new directories called "images" and "labels"
    Args:
        dataset_dir (str) : Directory containing the dataset.
        test_ratio (float) : Ratio of test samples.
        val_ratio (float) : Ratio of validation samples.
        seed (int) : Seed for reproducibility.

    """
    dataset_path = Path(dataset_dir).resolve()
    ann_dir = dataset_path / "ann"
    img_dir = dataset_path / "img"

    labels_out = dataset_path / "labels"
    images_out = dataset_path / "images"

    # 1. Recupera e mescola i file JSON con seed fisso
    json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    random.seed(seed)
    random.shuffle(json_files)

    # 2. Calcola lo split
    total = len(json_files)
    num_test = int(total * test_ratio)
    num_val = int(total * val_ratio)

    splits = {
        "test": json_files[:num_test],
        "val": json_files[num_test : num_test + num_val],
        "train": json_files[num_test + num_val :],
    }

    log(f"[LOG] Found {total} files", "blue")

    for split_name, files in splits.items():
        split_label_dir = labels_out / split_name
        split_img_dir = images_out / split_name

        split_label_dir.mkdir(parents=True, exist_ok=True)
        split_img_dir.mkdir(parents=True, exist_ok=True)

        for ann_file in files:
            # --- A. CONVERSIONE JSON -> YOLO TXT ---
            ann_path = ann_dir / ann_file
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_h = data.get("size", {}).get("height", 0)
            img_w = data.get("size", {}).get("width", 0)

            if img_h == 0 or img_w == 0:
                continue

            yolo_lines = []
            for obj in data.get("objects", []):
                cls_id = CLASS_MAPPING.get(obj.get("classId"))
                if cls_id is None:
                    continue

                p = obj.get("points", {}).get("exterior", [])
                if len(p) < 2:
                    continue

                x1, y1 = p[0]
                x2, y2 = p[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # Identifica il nome base dell'immagine
            base_name = Path(ann_file).stem
            if "." in base_name:
                base_name = Path(base_name).stem

            # Salva il file .txt nella relativa cartella dello split
            txt_path = split_label_dir / f"{base_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            # --- B. COPIA DELL'IMMAGINE ---
            # Cerca l'immagine originale corrispondente (supporta varie estensioni)
            found_img = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                candidate = img_dir / f"{base_name}{ext}"
                if candidate.exists():
                    found_img = candidate
                    break

            if found_img:
                shutil.copy(found_img, split_img_dir / found_img.name)

        log(f"[LOG] Split '{split_name}' completed [{len(files)} files].","green")

    log("[LOG] Dataset entirely prepared successfully!", "green")