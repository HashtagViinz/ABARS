import os
from pathlib import Path
from YOLO.model import yolo_model

def print_models_status():
    """
    Scans YOLO/tuning_results and YOLO/trained_model to print a CLI dashboard
    of all models in the yolo_model enum.
    """
    tuning_dir = Path("YOLO/tuning_results")
    trained_dir = Path("YOLO/trained_model")
    
    print("\n" + "="*65)
    print(f"| {'MODELLO':<20} | {'TUNING (Hyperparams)':<20} | {'TRAINING (Pesi)':<15} |")
    print("="*65)
    
    for model_enum in yolo_model:
        model_basename = model_enum.value.split('.')[0] # e.g. yolov8n
        
        # Check tuning status
        tune_status = "❌"
        if tuning_dir.exists():
            matching_dirs = list(tuning_dir.glob(f"tune_{model_basename}_*"))
            if matching_dirs:
                # Get latest
                latest_tune_dir = max(matching_dirs, key=os.path.getmtime)
                if (latest_tune_dir / "best_hyperparameters.yaml").exists():
                    tune_status = "✅"
        
        # Check trained status
        train_status = "❌"
        if trained_dir.exists():
            is_trained = False
            for run_dir in trained_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "weights" / "best.pt").exists():
                    args_file = run_dir / "args.yaml"
                    if args_file.exists():
                        try:
                            content = args_file.read_text()
                            if f"model: {model_enum.value}" in content:
                                is_trained = True
                                break
                        except Exception:
                            pass
            
            if is_trained:
                train_status = "✅"
                
        # Per far stampare correttamente i caratteri nel terminale allineati, usiamo formattazione standard
        print(f"| {model_enum.name:<20} | {tune_status:^19} | {train_status:^14} |")
        
    print("="*65 + "\n")
