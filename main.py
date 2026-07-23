from YOLO import tune_hyperparameters
from parser_command import command
from YOLO import yolo_model
from dataset import convert_json_to_yolo
import argparse
from logger import log


YAML_PATH = "dataset/uavod10.yaml"  # Path to the dataset YAML file



def arg_analyzer() -> None:
    """
    This function analyzes command-line arguments and returns them as a Namespace object.
    """

    parser = argparse.ArgumentParser(description="ABARS - Aereal Building Abuse Recognition System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ? Command to prepare the Dataset for training the YOLO model
    dataset_prepare = subparsers.add_parser("dataset_prepare", help="Prepare the dataset for YOLO training")
    
    
    # ? Command to test find the Hyperparameters for the YOLO model
    tune_hyperparameters = subparsers.add_parser("tune", help="Avvia il training del modello YOLO")
    tune_hyperparameters.add_argument(
        "--model",
        type=lambda s: yolo_model[s.upper()],   # Parse the model argument to the corresponding yolo_model enum
        required=True,
        choices=list(yolo_model),
        help="Syze of the Model ('NANO', 'SMALL', 'MEDIUM')",
    )   
    
    args = parser.parse_args()


    # ? TUNE COMMAND
    if args.command == command.DATASET_PREPARE.value:
        convert_json_to_yolo(
            "dataset/ann/",
            "dataset/label/"
        )
        
    
    if args.command == command.TUNE.value:
        
        tune_hyperparameters(
            model_enum=args.model,
            epochs=30,
            iterations=30,
            optimizer="auto",
            yaml_path=YAML_PATH
        )
        pass


if __name__ == "__main__":
    arg_analyzer()
