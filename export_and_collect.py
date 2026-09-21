import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from logger import log

TOP_MODELS = [
    "v26s_untuned_tiled_bilateral_d11",
    "v26s_untuned_tiled-3",
    "v26s_untuned_dehaze_bilateral_sharp",
    "v26m_untuned_tiled_bilateral_sharp",
    "v26m_untuned_tiled-6",
    "v26n_untuned_tiled_bilateral_d11_sharp",
    "v26n_untuned_tiled-2"
]

def main():
    base_dir = Path("YOLO/trained_model")
    report_dir = Path("report_graphics")
    
    os.makedirs(report_dir, exist_ok=True)
    
    for model_name in TOP_MODELS:
        model_dir = base_dir / model_name
        weights_path = model_dir / "weights" / "best.pt"
        
        if not weights_path.exists():
            log(f"Cartella o pesi 'best.pt' non trovati per {model_name} in {weights_path}. Salto...", "red")
            continue
            
        log(f"--- Processando {model_name} ---", "blue")
        
        # 1. Copia Grafici per la Relazione
        target_gfx = report_dir / model_name
        os.makedirs(target_gfx, exist_ok=True)
        
        png_files = list(model_dir.glob("*.png")) + list(model_dir.glob("*.jpg"))
        if png_files:
            log(f"Trovati {len(png_files)} grafici, copiatura in {target_gfx}...", "cyan")
            for pf in png_files:
                shutil.copy(pf, target_gfx / pf.name)
        else:
            log(f"Nessun grafico trovato in {model_dir}", "yellow")
            
        # 2. Esportazione verso formato NCNN
        log(f"Avvio esportazione NCNN per {model_name}...", "yellow")
        try:
            model = YOLO(str(weights_path))
            # format="ncnn" converte il modello, imgsz=1024 assicura che usi la risoluzione del nostro train
            model.export(format="ncnn", imgsz=1024)
            log(f"Esportazione NCNN completata con successo per {model_name}!", "green")
        except Exception as e:
            log(f"Errore durante esportazione di {model_name}: {e}", "red")

if __name__ == "__main__":
    main()
