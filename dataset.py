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
    


def convert_json_to_yolo(ann_dir: str, labels_out: str) -> None:
    """
    Converts JSON annotations to YOLO format.
    Args: 
        ann_dir (str) : Directory containing the JSON annotation files.
        labels_out (str) : Directory where the YOLO formatted label files will be saved.

    """
    
    os.makedirs(labels_out, exist_ok=True)
    log(f"[LOG] Starting conversion {labels_out}...", "blue")

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'): 
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Dimensioni immagine per normalizzazione
        img_h = data.get('size', {}).get('height', 0)
        img_w = data.get('size', {}).get('width', 0)

        if img_h == 0 or img_w == 0:
            log(f"[WARN] Not valid size {ann_file}, ignored.", "yellow")
            continue

        yolo_lines = []
        for obj in data.get('objects', []):
            cls_id = CLASS_MAPPING.get(obj.get('classId'))
            if cls_id is None: 
                continue
            
            # Punti exterior: [[x1, y1], [x2, y2]]
            p = obj.get('points', {}).get('exterior', [])
            if len(p) < 2:
                continue

            x1, y1 = p[0]
            x2, y2 = p[1]
            
            # Coordinate bounding box pulite
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            # Normalizzazione (0 - 1)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            # Formattazione float a 6 cifre decimali per pulizia
            yolo_lines.append(f"{cls_id} {x_center} {y_center} {w} {h}")

        # Generazione nome file .txt compatibile
        # Es: 'frame_001.jpg.json' -> 'frame_001.txt'
        base_name = Path(ann_file).stem  # rimuove .json
        if '.' in base_name: 
            base_name = Path(base_name).stem # rimuove eventuale .jpg / .png
            
        txt_path = os.path.join(labels_out, f"{base_name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

    log("[LOG] All the Label are created successfully", "green")
    
    
    
    
if __name__ == "__main__":
    # Inserisci i tuoi percorsi reali qui
    img_dir = "dataset/images/train/"
    ann_dir = "dataset/ann/"
    
    # 1. DEFINIAMO LE TRASFORMAZIONI
    # Usiamo ColorJitter per alterare i colori e la luminosità
    my_transforms = v2.Compose([
        v2.ToImage(),  # Converte l'immagine PIL in Tensore (sostituisce F.to_tensor)
        v2.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2), # Applica filtri colore
        v2.ToDtype(torch.float32, scale=True) # Scala i valori da 0-255 a 0.0-1.0
    ])
    
    # 2. CREIAMO DUE ISTANZE DEL DATASET
    # Una "liscia" e una con le trasformazioni applicate
    dataset_original = UAVOD(images_dir=img_dir, annotations_dir=ann_dir, transforms=None)
    dataset_transformed = UAVOD(images_dir=img_dir, annotations_dir=ann_dir, transforms=my_transforms)
    
    # 3. ESTRAIAMO LA STESSA IMMAGINE (Indice 0) DA ENTRAMBI
    img_orig, target_orig = dataset_original[0]
    img_trans, target_trans = dataset_transformed[0]
    
    # Inverto il dict per avere da indice a nome classe
    idx_to_class = {v: k for k, v in dataset_original.class_map.items()}
    
    # --- ELABORAZIONE IMMAGINE ORIGINALE ---
    img_orig_uint8 = (img_orig * 255).to(torch.uint8)
    labels_orig = [idx_to_class[l.item()] for l in target_orig["labels"]]
    res_orig = draw_bounding_boxes(
        img_orig_uint8, target_orig["boxes"], labels=labels_orig, colors="red", width=3
    )
    
    # --- ELABORAZIONE IMMAGINE TRASFORMATA ---
    # Nota: Moltiplichiamo per 255 perché v2.ToDtype l'ha portata nel range 0.0 - 1.0
    img_trans_uint8 = (img_trans * 255).to(torch.uint8) 
    labels_trans = [idx_to_class[l.item()] for l in target_trans["labels"]]
    res_trans = draw_bounding_boxes(
        img_trans_uint8, target_trans["boxes"], labels=labels_trans, colors="blue", width=3
    )
    
    # --- VISUALIZZAZIONE AFFIANCATA CON MATPLOTLIB ---
    # Creiamo una figura con 1 riga e 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Disegniamo l'originale a sinistra (axes[0])
    axes[0].imshow(res_orig.permute(1, 2, 0).numpy())
    axes[0].set_title("Originale (Nessuna Trasformazione)", fontsize=14)
    axes[0].axis("off")
    
    # Disegniamo la trasformata a destra (axes[1])
    axes[1].imshow(res_trans.permute(1, 2, 0).numpy())
    axes[1].set_title("Trasformata (ColorJitter)", fontsize=14)
    axes[1].axis("off")
    
    # Mostriamo il risultato
    plt.tight_layout()
    plt.show()

