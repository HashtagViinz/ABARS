from YOLO import tune_hyperparameters, is_GPUs_available
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
    tune_hyperparameters_command = subparsers.add_parser("tune", help="Avvia il training del modello YOLO")
    tune_hyperparameters_command.add_argument(
        "--model",
        type=lambda s: yolo_model[s.upper()],   # Parse the model argument to the corresponding yolo_model enum
        required=True,
        choices=list(yolo_model),
        help="Syze of the Model ('NANO', 'SMALL', 'MEDIUM')",
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


if __name__ == "__main__":
    arg_analyzer()
