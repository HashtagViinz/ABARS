import albumentations as A
import ultralytics.data.augment as aug
from logger import log

def inject_custom_albumentations(filter_type: str):
    if filter_type == "none":
        return

    log(f"Injecting custom Albumentations pipeline: {filter_type}", "yellow")

    transforms = []
    
    if filter_type in ["clahe", "all"]:
        # Applichiamo CLAHE sempre (p=1.0)
        transforms.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0))
        
    if filter_type in ["sharpen", "all"]:
        transforms.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0))
        
    if filter_type in ["white_balance", "all"]:
        # ColorJitter approssima il white balance e aggiusta luminosita'/contrasto
        transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0))
        
    if filter_type in ["grayscale", "all"]:
        # Forza la conversione in scala di grigi
        transforms.append(A.ToGray(p=1.0))

    # Conserviamo il costruttore originale in caso serva
    original_init = aug.Albumentations.__init__
    
    def custom_init(self, p=1.0, *args, **kwargs):
        # Ignoriamo i transform di default di YOLO passandogli i nostri
        original_init(self, p=1.0, transforms=transforms)
        
    # Applichiamo l'hack
    aug.Albumentations.__init__ = custom_init
    log("Albumentations hook applied successfully!", "green")
