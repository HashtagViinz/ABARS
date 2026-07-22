def log(text: str,color:str = None) -> None:
    """
    Stampa un messaggio di log con un prefisso standard.

    Args:
        text (str): Il messaggio da loggare.
        color (str): Il colore del messaggio. Defaults to None.
    """
    if color:
        print(f"{colorize(text, color)}")
    else:
        print(f"{text}")


def colorize(text: str, color: str) -> str:
    """
    Colora una stringa con i codici ANSI per il terminale.

    Args:
        text (str): La stringa da colorare.
        color (str): Il colore (es: "red", "green", "yellow", "blue", "magenta", "cyan", "white").

    Returns:
        str: La stringa colorata con codici ANSI.
    """
    colors = {
        "black": "\033[30m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
        "orange": "\033[33m"
    }

    start = colors.get(color.lower(), "")
    end = colors["reset"] if start else ""
    return f"{start}{text}{end}"