import os
import shutil
from pathlib import Path
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
    target_dir = Path("ncnn_models_ready")
    
    os.makedirs(target_dir, exist_ok=True)
    log(f"Creata cartella di destinazione: {target_dir}", "blue")
    
    for model_name in TOP_MODELS:
        ncnn_dir = base_dir / model_name / "weights" / "best_ncnn_model"
        
        if not ncnn_dir.exists():
            log(f"Modello NCNN non trovato per {model_name} in {ncnn_dir}. Sicuro di aver lanciato l'export?", "red")
            continue
            
        target_model_dir = target_dir / f"{model_name}_ncnn"
        
        if target_model_dir.exists():
            shutil.rmtree(target_model_dir)
            
        shutil.copytree(ncnn_dir, target_model_dir)
        log(f"Copiato modello NCNN -> {target_model_dir}", "green")
        
    log("Tutti i modelli sono stati raggruppati in 'ncnn_models_ready'!", "cyan")

if __name__ == "__main__":
    main()
