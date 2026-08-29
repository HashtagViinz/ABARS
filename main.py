from YOLO import tune_hyperparameters, is_GPUs_available,train_model
from parser_command import command
from YOLO import yolo_model
from dataset import process_and_split_dataset
import argparse
from logger import log


YAML_PATH = "dataset/uavod10.yaml"  # Path to the dataset YAML file


def arg_analyzer() -> None:
    """
    This function analyzes command-line arguments and returns them as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="ABARS - Aereal Building Abuse Recognition System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1) Command to prepare the Dataset for training the YOLO model
    dataset_prepare_command = subparsers.add_parser("dataset_prepare", help="Prepare the dataset for YOLO training")
    
    # 2) Test Hardware Node
    hardware_test = subparsers.add_parser("hw_test", help="Test hardware node")

    # 3) Command to test find the Hyperparameters for the YOLO model
    tune_hyperparameters_command = subparsers.add_parser("tune", help="Start the tuning for the YOLO Model")
    tune_hyperparameters_command.add_argument(
        "--model",
        type=lambda s: yolo_model[s.upper()],   # Parse the model argument to the corresponding yolo_model enum
        required=True,
        choices=list(yolo_model),
        help="Syze of the Model ('NANO', 'SMALL', 'MEDIUM')",
    )

    # 4) Command to train the YOLO model
    train_command = subparsers.add_parser("train", help="Start the Train for the YOLO Model")
    train_command.add_argument(
        "--model",
        type=lambda s: yolo_model[s.upper()],   # Parse the model argument to the corresponding yolo_model enum
        required=True,
        choices=list(yolo_model),
        help="Syze of the Model ('NANO', 'SMALL', 'MEDIUM')",
    )
    train_command.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs for training",
    )
    train_command.add_argument(
        "--patience",
        type=int,
        required=True,
        help="Number of epochs with no improvement",
    )
    train_command.add_argument(
        "--name",
        type=str,
        required=False,
        help="Top Level name to identify the Model.",
    )
    train_command.add_argument(
        "--baseline",
        action="store_true",
        help="Ignore tuned hyperparameters and use YOLO defaults for a baseline run.",
    )
    train_command.add_argument(
        "--dataset",
        type=str,
        default="dataset/uavod10.yaml",
        help="Percorso al file yaml del dataset (default: dataset/uavod10.yaml)."
    )
    train_command.add_argument(
        "--filter",
        nargs="+",
        type=str,
        default=["none"],
        help="Lista di filtri da applicare (es. --filter clahe grayscale)."
    )
    train_command.add_argument(
        "--tile",
        action="store_true",
        help="Applica lo slicing offline 2x2 (Offline Tiling) al dataset prima del training."
    )

    # 4.5) Command to generate static filtered dataset
    generate_command = subparsers.add_parser("generate_dataset", help="Genera una copia fisica del dataset applicando un filtro CV o tiling.")
    generate_command.add_argument(
        "--filter",
        nargs="+",
        type=str,
        default=["none"],
        help="Lista di filtri da applicare (es. --filter clahe grayscale)."
    )
    generate_command.add_argument(
        "--tile",
        action="store_true",
        help="Applica lo slicing offline 2x2 (Offline Tiling) al dataset."
    )

    # 5) Command to test MLflow
    mlflow_test_command = subparsers.add_parser("mlflow_test", help="Test MLflow logging and connection")

    args = parser.parse_args()


    # ? DATASET PREPARE COMMAND
    if args.command == command.DATASET_PREPARE.value:
        """
        This command prepare the Dataset in order to be 
        used by YOLO Models
        """
        process_and_split_dataset("dataset",test_ratio=0.1,val_ratio=0.2,seed=42)
    # ? HW TEST COMMAND
    elif args.command == command.HW_TEST.value:
        """
        This command test the hardware node
        """
        device = is_GPUs_available()
    # ? MLFLOW TEST COMMAND
    elif args.command == command.MLFLOW_TEST.value:
        from YOLO.model import test_mlflow
        test_mlflow()
    # ? TUNE COMMAND
    elif args.command == command.TUNE.value:
        """
        This command try to understands the best hyperparams
        based on hardware of our machine
        """
        tune_hyperparameters(
            model_enum=args.model,
            epochs=30,
            iterations=30,
            optimizer="auto",
            yaml_path=YAML_PATH
        )

    elif args.command == command.TRAIN.value:
        """
        This command train the YOLO model based on the best hyperparameters
        found during the tuning process.
        """
        import os
        model_name = args.model.value.split('.')[0]
        dynamic_cfg_path = f"YOLO/tuning_results/tune_{model_name}_E30_Iter30/best_hyperparameters.yaml"
        
        if not args.baseline and os.path.exists(dynamic_cfg_path):
            log(f"[INFO] Uso gli iperparametri custom da: {dynamic_cfg_path}", "blue")
            cfg_path_to_use = dynamic_cfg_path
        else:
            if args.baseline:
                log(f"[INFO] Modalità BASELINE forzata. Uso i default di YOLO ignorando il file di tuning.", "yellow")
            else:
                log(f"[INFO] Nessun file di tuning trovato per {model_name}. Uso i default di YOLO.", "yellow")
            cfg_path_to_use = None

        # Se c'è un filtro, generiamo il dataset invisibilmente e aggiorniamo lo yaml!
        dataset_path = args.dataset
        if "none" not in args.filter:
            log(f"Rilevati filtri in input: {args.filter}. Avvio il dataloader generativo...", "yellow")
            from cv.dataset_builder import build_filtered_dataset
            # args.filter ora è una lista, es. ["clahe", "grayscale"]
            dataset_path = build_filtered_dataset(dataset_path, args.filter)
            log(f"YOLO ricevera' il dataset filtrato da: {dataset_path}", "green")
        else:
            log("Nessun filtro rilevato, procedo col dataset standard a colori.", "yellow")
            
        if args.tile:
            log("Rilevata richiesta Tiling. Avvio lo slicing del dataset...", "yellow")
            from cv.tiler import build_tiled_dataset
            dataset_path = build_tiled_dataset(dataset_path)
            log(f"YOLO ricevera' il dataset piastrellato da: {dataset_path}", "green")

        train_model(
            model=args.model,
            epochs=args.epochs,
            btch_size=32,
            img_size=1024,
            patience=args.patience,
            yaml_path=dataset_path,
            cfg_path=cfg_path_to_use,
            name=args.name if args.name is not None else None
        )

    elif args.command == "generate_dataset":
        dataset_path = YAML_PATH
        if "none" not in args.filter:
            from cv.dataset_builder import build_filtered_dataset
            dataset_path = build_filtered_dataset(dataset_path, args.filter)
        if args.tile:
            from cv.tiler import build_tiled_dataset
            dataset_path = build_tiled_dataset(dataset_path)
        log(f"Dataset finale generato in: {dataset_path}", "green")

    
        
if __name__ == "__main__":
    arg_analyzer()
