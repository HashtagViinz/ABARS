import os
import yaml
from pathlib import Path
from logger import log
from collections import defaultdict

def analyze_dataset_stats(yaml_path: str):
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        log(f"Errore: Il file {yaml_path} non esiste.", "red")
        return
        
    with open(yaml_file, 'r') as f:
        dataset_yaml = yaml.safe_load(f)
        
    # Risoluzione percorsi relativi al yaml
    dataset_dir = yaml_file.parent
    if 'path' in dataset_yaml:
        # Se 'path' è definito, i path train/val sono relativi a quel 'path'
        base_dir = Path(dataset_yaml['path'])
        if not base_dir.is_absolute():
            base_dir = dataset_dir / base_dir
    else:
        base_dir = dataset_dir
        
    train_dir = base_dir / dataset_yaml.get('train', 'train')
    val_dir = base_dir / dataset_yaml.get('val', 'val')
    
    classes = dataset_yaml.get('names', {})
    if isinstance(classes, list):
        classes = {i: name for i, name in enumerate(classes)}
        
    log(f"Analisi del dataset: {yaml_path}", "blue")
    
    def analyze_split(split_dir: Path):
        images_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir
        labels_dir = split_dir / 'labels' if (split_dir / 'labels').exists() else split_dir.parent / 'labels' / split_dir.name
        
        # Caso in cui la struttura sia dataset/images/train e dataset/labels/train
        if not images_dir.exists():
             images_dir = base_dir / 'images' / split_dir.name
        if not labels_dir.exists():
             labels_dir = base_dir / 'labels' / split_dir.name
             
        if not images_dir.exists() or not labels_dir.exists():
            log(f"Non riesco a trovare le cartelle images/labels per {split_dir}", "yellow")
            return 0, 0, {}
            
        total_images = 0
        background_images = 0
        class_counts = defaultdict(int)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                total_images += 1
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                is_background = True
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            is_background = False
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) > 0:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    
                if is_background:
                    background_images += 1
                    
        return total_images, background_images, class_counts

    log("Scansione Train set...", "cyan")
    t_imgs, t_bgs, t_counts = analyze_split(Path(dataset_yaml.get('train', 'train')))
    
    log("Scansione Validation set...", "cyan")
    v_imgs, v_bgs, v_counts = analyze_split(Path(dataset_yaml.get('val', 'val')))
    
    tot_imgs = t_imgs + v_imgs
    tot_bgs = t_bgs + v_bgs
    
    print("\n" + "="*50)
    print("📊 DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Totale Immagini:      {tot_imgs} (Train: {t_imgs}, Val: {v_imgs})")
    
    bg_perc = (tot_bgs / tot_imgs * 100) if tot_imgs > 0 else 0
    bg_color = "red" if bg_perc > 15 else "green"
    log(f"Immagini Background:  {tot_bgs} ({bg_perc:.2f}%)", bg_color)
    
    if bg_perc > 15:
        log("⚠️ ATTENZIONE: Il dataset contiene più del 15% di immagini di background.", "yellow")
        log("Questo potrebbe rallentare l'addestramento o sbilanciare il modello. Valuta di potare i background.", "yellow")
    elif bg_perc == 0:
         log("⚠️ ATTENZIONE: Il dataset NON contiene background. È fortemente raccomandato avere uno 0-10% di background per ridurre i Falsi Positivi.", "yellow")
        
    print("-" * 50)
    print("DISTRIBUZIONE CLASSI (Train + Val):")
    
    tot_instances = 0
    for cls_id in classes.keys():
        tot_instances += t_counts.get(cls_id, 0) + v_counts.get(cls_id, 0)
        
    if tot_instances == 0:
        print("Nessuna istanza trovata (dataset vuoto o path errati).")
    else:
        for cls_id, cls_name in classes.items():
            count = t_counts.get(cls_id, 0) + v_counts.get(cls_id, 0)
            perc = (count / tot_instances) * 100
            print(f" - [{cls_id}] {cls_name:<15}: {count:<6} istanze ({perc:.1f}%)")
    print("="*50 + "\n")
