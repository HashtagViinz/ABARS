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
        
        if os.path.exists(dynamic_cfg_path):
            log(f"[INFO] Uso gli iperparametri custom da: {dynamic_cfg_path}", "blue")
            cfg_path_to_use = dynamic_cfg_path
        else:
            log(f"[INFO] Nessun file di tuning trovato per {model_name}. Uso i default di YOLO.", "yellow")
            cfg_path_to_use = None

        train_model(
            model = args.model,
            epochs = args.epochs,
            patience = args.patience,
            btch_size = 32,
            img_size = 1024,
            yaml_path = YAML_PATH,
            cfg_path = cfg_path_to_use,
            name = args.name if args.name is not None else None
        )

    
    

if __name__ == "__main__":
    arg_analyzer()
