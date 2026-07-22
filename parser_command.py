from enum import StrEnum


class command(StrEnum): 
    """
    command Enum for the ABARS CLI commands.
    """
    TUNE = "tune"
    DATASET_PREPARE = "dataset_prepare"

