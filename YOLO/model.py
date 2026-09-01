from enum import StrEnum
import os
import torch
from ultralytics import YOLO, settings
from logger import log
import mlflow

# ! COSTANTS
YOLO_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the current file
MLFLOW_EXPERIMENT_NAME = "ABARS_YOLO"  # Name of the MLflow experiment

# Configura Ultralytics per scaricare i modelli base in YOLO/base_models
BASE_MODELS_DIR = os.path.join(YOLO_DIR, 'base_models')
os.makedirs(BASE_MODELS_DIR, exist_ok=True)
settings.update({'weights_dir': BASE_MODELS_DIR})

class yolo_model(StrEnum): 
    """
    Enum dei modelli YOLO disponibili per il training.
    """
    V8_NANO = "yolov8n.pt"
    V8_SMALL = "yolov8s.pt"
    V8_MEDIUM = "yolov8m.pt"
    V11_NANO = "yolo11n.pt"
    V11_SMALL = "yolo11s.pt"
    V11_MEDIUM = "yolo11m.pt"

def is_GPUs_available() -> str:
    """
    This function checks the available hardware (GPUs).
    
    Returns:
        str: A string representing the available GPUs. If no GPUs are available, it returns 'cpu'.
    """
    log("Checking HW : ","yellow")
    device_arg = 'cpu'  # Default to CPU if no GPUs are available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            log(f"- GPU ID {i}: {gpu_name}", "yellow")
        device_arg = ",".join(str(i) for i in range(gpu_count))
    else:
        log("Using CPU", "yellow")
    return device_arg


def on_epoch_end_callback(trainer) -> None:
    """
    Callback executed by Ultralytics at the end of each epoch.
    Prints a summary of the metrics:
 
    - Current epoch / total epochs
    - Current Learning Rate
    - gpu memory (VRAM) used
    - Loss di training (media dell'epoca corrente)
    - mAP@50 e mAP@50-95 di validazione
    
    """
    current_epoch = trainer.epoch + 1
    total_epochs = trainer.epochs

    # --- METRICHE DI VALIDAZIONE ---
    metrics = getattr(trainer.validator, 'metrics', None)
    map50 = metrics.box.map50 if metrics and hasattr(metrics, 'box') else 0.0
    map50_95 = metrics.box.map if metrics and hasattr(metrics, 'box') else 0.0

    # --- METRICHE DI LOSS (Training & Validation) ---
    # Loss di Training (media dell'epoca corrente)
    train_loss = trainer.tloss.tolist() if hasattr(trainer, 'tloss') and trainer.tloss is not None else []
    
    # --- LEARNING RATE CORRENTE ---
    current_lr = trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else 0.0

    # --- MONITORAGGIO VRAM (se su GPU) ---
    gpu_memory = ""
    if torch.cuda.is_available():
        # Restituisce la memoria allocata sulla GPU principale in GB
        mem_gb = torch.cuda.memory_reserved() / 1E9
        gpu_memory = f" | VRAM: {mem_gb:.2f} GB"

    # --- STAMPA DEL LOG ---
    log(f"--- EPOCH : {current_epoch}/{total_epochs} GPU_MEM: {gpu_memory} ---", "green")
    log(f"   >> Learning Rate  : {current_lr:.6f}", "cyan")
    
    if train_loss:
        # Se tloss ha 3 elementi (box_loss, cls_loss, dfl_loss)
        log(f"   >> Train Loss     : {sum(train_loss):.4f}", "yellow")
        
    log(f"   >> Val mAP@50     : {map50:.4f}", "yellow")
    log(f"   >> Val mAP@50-95  : {map50_95:.4f}", "yellow")
    log("------------------------------------------", "green")






def tune_hyperparameters(
    model_enum: yolo_model,
    epochs: int = 30,
    iterations: int = 30,
    optimizer: str = "auto",
    yaml_path: str = None
) -> None:
    """
    It searches for optimal hyperparameters (lr0, batch, momentum, etc.) 
    using the genetic algorithms built into Ultralytics YOLO.
    
    Args:
        model_enum (yolo_model): Enum del modello YOLO da ottimizzare.
        epochs (int): Numero di epoche per ogni singolo test/iterazione.
        iterations (int): Numero di combinazioni/esperimenti da testare.
        optimizer (str): Ottimizzatore da testare (es. 'AdamW', 'SGD', 'auto').
        yaml_path (str): Percorso del file del dataset YAML.
    """
    devices = is_GPUs_available()
    
    if not os.path.exists(yaml_path):
        log(f"Dataset YAML file not found at: {yaml_path}", "red")
        return
    
    tuning_dir = os.path.join(YOLO_DIR, 'tuning_results')
    model_name = model_enum.value.split('.')[0]
    log_file = f"tune_{model_name}_E{epochs}_Iter{iterations}"
    log(f"START TUNING - {model_name} ", "cyan")
    
    model = YOLO(model_enum.value)
    

    model.tune(
        data=yaml_path,
        epochs=epochs,
        iterations=iterations,
        optimizer=optimizer,
        device=devices,
        project=tuning_dir,
        name=log_file,
        workers=8,
        plots=True,
        save=True
    )

    log(f"Tuning completed! Optimal hyperparameters saved in: {tuning_dir}/{log_file}", "green")
 
 
def test_mlflow() -> None:
    """
    Function to test MLflow connection and logging.
    """
    log("Testing MLflow connection...", "yellow")
    yolo_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        project_root = os.path.dirname(yolo_dir)
        db_path = os.path.join(project_root, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        log(f"Using local DB: sqlite:///{db_path}", "cyan")
    else:
        log(f"Using MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}", "cyan")
        
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME + "_TEST")
    
    with mlflow.start_run(run_name="test_connection"):
        mlflow.log_param("test_param", "successful")
        mlflow.log_metric("test_metric", 1.0)
        log("Logged test param and metric successfully.", "green")
        
    log("MLflow test completed! Check your DB or UI.", "green")
 
 

def train_model(model: yolo_model, epochs: int, btch_size:int, img_size:int, patience: int, yaml_path: str, cfg_path: str, name:str = None) -> None:
    """
    Function to train the YOLO model.
    Is Setupped to log training metrics using MLflow and the on_epoch_end_callback.
    
    Args:
        model (yolo_model): The YOLO model to be trained.
        epochs (int): Number of training epochs.
        btch_size (int): Batch size for training.
        img_size (int): Image size for training.
        patience (int): Number of epochs with no improvement.
        yaml_path (str): Path to the dataset YAML file.
        cfg_path (str): Path to use the best hyperprameters YAML file.
        name (str): Top Level name to identify the Model.
    """
    
    model_name = model.value.split('.')[0]  # Extract model name without extension
    
    # Configure MLflow variables BEFORE loading the YOLO model
    # MLflow trackig uri logic setup
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    
    if mlflow_tracking_uri:
        # User defined an external mlflow server (SLURM/Cluster)
        log(f"Using external MLflow server: {mlflow_tracking_uri}", "blue")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    else:
        # Default local sqlite database
        db_path = os.path.join(os.getcwd(), 'mlflow.db')
        log(f"Using local MLflow database: {db_path}", "blue")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)     #Setup MLflow experiment 
    
    yolo_model_instance = YOLO(model.value)       # Load the specified YOLO model
    yolo_model_instance.add_callback("on_epoch_end", on_epoch_end_callback)   # Add the Callback for logging
    
    yolo_dir = os.path.dirname(os.path.abspath(__file__))
    trained_models_dir = os.path.join(yolo_dir, 'trained_model')
    devices = is_GPUs_available()   # Check device availability (GPU or CPU)
    
    # ! Starting the training process
    results = yolo_model_instance.train(
        data=yaml_path,
        cfg=cfg_path,
        epochs=epochs,      
        imgsz=img_size,         
        batch=btch_size,                       
        device=devices,       
        name=name if name else model_name,
        project=trained_models_dir,
        patience=patience,
        workers=8,
        save=True,
        save_period=1
    )    
    
    log(f"Training Done. Result in : {trained_models_dir}", "green")

    
    
    
        
    
    
    
    