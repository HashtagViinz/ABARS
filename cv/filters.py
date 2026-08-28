import cv2
import numpy as np
from abc import ABC, abstractmethod

class BaseFilter(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applica il filtro all'immagine.
        L'immagine in input deve essere in formato BGR (default di OpenCV).
        """
        pass

class CLAHEFilter(BaseFilter):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convertiamo in spazio colore LAB per applicare il CLAHE solo alla Luminosita' (L)
        # In questo modo non falsiamo i colori originali (A, B)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Applichiamo il CLAHE alla luminanza
        cl = self.clahe.apply(l)
        
        # Riuniamo i canali e torniamo in BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

class GrayscaleFilter(BaseFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        # Converte in bianco e nero, ma lo riporta in formato a 3 canali 
        # (YOLO richiede sempre tensori a 3 canali)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

class CVPipeline:
    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    def process(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        for f in self.filters:
            result = f.apply(result)
        return result
