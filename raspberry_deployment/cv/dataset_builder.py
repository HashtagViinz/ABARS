import os
import cv2
import shutil
from pathlib import Path
from logger import log
from cv.filters import CLAHEFilter, GrayscaleFilter, CVPipeline

def build_filtered_dataset(source_yaml: str, filters_list: list[str]):
    """
    Legge il dataset originale, applica i filtri alle immagini e crea 
    una copia fisica del dataset (con label copiate e un nuovo yaml).
    """
    # Definisci la pipeline dinamicamente in base alla lista di filtri passata
    filters = []
    for filter_name in filters_list:
        if filter_name == "clahe":
            filters.append(CLAHEFilter())
        elif filter_name == "grayscale":
            filters.append(GrayscaleFilter())
        elif filter_name == "structural":
            from cv.filters import StructuralEdgeFilter
            filters.append(StructuralEdgeFilter())
        elif filter_name == "bilateral":
            from cv.filters import BilateralFilter
            filters.append(BilateralFilter())
        elif filter_name == "bilateral_sharp":
            from cv.filters import BilateralSharpenFilter
            filters.append(BilateralSharpenFilter())
        elif filter_name == "gamma":
            from cv.filters import GammaFilter
            filters.append(GammaFilter())
        elif filter_name == "hsv_boost":
            from cv.filters import HSVSaturationFilter
            filters.append(HSVSaturationFilter())
        elif filter_name == "dehaze":
            from cv.filters import DehazeFilter
            filters.append(DehazeFilter())
        else:
            log(f"Filtro {filter_name} non supportato, ignorato.", "red")
            
    if not filters:
        return source_yaml
        
    pipeline = CVPipeline(filters)
    
    # Setup percorsi - Il nome della cartella conterrà tutti i filtri in ordine
    combo_name = "_".join(filters_list)
    source_dir = Path("dataset")
    target_dir = Path(f"dataset_{combo_name}")
    new_yaml_path = target_dir / f"uavod10_{combo_name}.yaml"
    
    if target_dir.exists() and new_yaml_path.exists():
        log(f"Il dataset {target_dir} esiste gia'. Nessuna generazione necessaria.", "yellow")
        return new_yaml_path
        
    log(f"Inizio generazione del dataset modificato: {target_dir}", "blue")
    
    # Creiamo le cartelle per images e labels (train, val, test)
    for split in ["train", "val", "test"]:
        os.makedirs(target_dir / "images" / split, exist_ok=True)
        os.makedirs(target_dir / "labels" / split, exist_ok=True)
        
        src_img_dir = source_dir / "images" / split
        src_lbl_dir = source_dir / "labels" / split
        
        if not src_img_dir.exists():
            continue
            
        images = list(src_img_dir.glob("*.jpg"))
        total_imgs = len(images)
        log(f"Processando {total_imgs} immagini per lo split '{split}'...", "cyan")
        
        for idx, img_path in enumerate(images):
            # 1. Applica filtro all'immagine
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            processed_img = pipeline.process(img)
            
            # 2. Salva l'immagine processata
            target_img_path = target_dir / "images" / split / img_path.name
            cv2.imwrite(str(target_img_path), processed_img)
            
            # 3. Copia il file .txt delle etichette (senza modificarlo)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                target_lbl_path = target_dir / "labels" / split / lbl_path.name
                shutil.copy(lbl_path, target_lbl_path)
            
            # Logga il progresso ogni 100 immagini
            if (idx + 1) % 100 == 0 or (idx + 1) == total_imgs:
                log(f"[{split}] Progresso: {idx + 1}/{total_imgs} immagini elaborate", "blue")
                
    # Crea il nuovo file yaml
    # new_yaml_path already defined
    with open(new_yaml_path, "w") as f:
        # Percorsi assoluti richiesti da YOLO per evitare confusioni
        abs_target = target_dir.absolute()
        f.write(f"path: {abs_target}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        
        f.write("names:\n")
        classes = ['building', 'cable-tower', 'cultivation-mesh-cage', 'landslide', 'pool', 'prefabricated-house', 'quarry', 'ship', 'vehicle', 'well']
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")
            
    log(f"Generazione completata! Dataset pronto in {target_dir}", "green")
    return new_yaml_path

if __name__ == "__main__":
    build_filtered_dataset("dataset/uavod10.yaml", "clahe")
