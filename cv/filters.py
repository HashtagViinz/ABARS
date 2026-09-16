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

class BilateralFilter(BaseFilter):
    def __init__(self, d=11, sigmaColor=85, sigmaSpace=85):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Applica il filtro bilaterale direttamente sull'immagine BGR a colori
        return cv2.bilateralFilter(image, self.d, self.sigmaColor, self.sigmaSpace)

class StructuralEdgeFilter(BaseFilter):
    def __init__(self, clahe_clip=4.0, clahe_grid=(8, 8), 
                 bilateral_d=9, bilateral_sigmaColor=75, bilateral_sigmaSpace=75,
                 canny_th1=50, canny_th2=150):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.d = bilateral_d
        self.sigmaColor = bilateral_sigmaColor
        self.sigmaSpace = bilateral_sigmaSpace
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        
        # Kernel per le operazioni morfologiche
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def apply(self, image: np.ndarray) -> np.ndarray:
        # 1. Spazio LAB -> Separazione Luminanza
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Equalizzazione locale del contrasto (CLAHE)
        l_clahe = self.clahe.apply(l)
        
        # 3. Filtraggio non-lineare edge-preserving (Bilateral)
        l_blur = cv2.bilateralFilter(l_clahe, self.d, self.sigmaColor, self.sigmaSpace)
        
        # 4. Rilevamento bordi (Canny)
        edges = cv2.Canny(l_blur, self.canny_th1, self.canny_th2)
        
        # 5. Consolidamento morfologico
        # Chiusura (collega i bordi spezzati dei muri)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel_close)
        # Apertura (elimina il "rumore di sale" spurio isolato)
        edges_clean = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, self.kernel_open)
        
        # Riportiamo a 3 canali perché YOLO vuole un input (H, W, 3)
        return cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)


class CVPipeline:
    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    def process(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        for f in self.filters:
            result = f.apply(result)
        return result
