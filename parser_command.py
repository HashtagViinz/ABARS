from enum import StrEnum


class command(StrEnum): 
    """
    command Enum for the ABARS CLI commands.
    """
    TUNE = "tune"
    HW_TEST = "hw_test"
    DATASET_PREPARE = "dataset_prepare"
    TRAIN = "train"
    MLFLOW_TEST = "mlflow_test"
    STATUS = "status"

