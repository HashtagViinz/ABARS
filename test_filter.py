import sys
import cv2
import numpy as np
from pathlib import Path
from cv.filters import CLAHEFilter, GrayscaleFilter, StructuralEdgeFilter, CVPipeline
from logger import log

def main():
    if len(sys.argv) < 3:
        print("Uso: uv run python test_filter.py <percorso_immagine> <nome_filtro>")
        print("Esempio: uv run python test_filter.py dataset/images/train/000000000009.jpg structural")
        sys.exit(1)

    img_path = sys.argv[1]
    filter_name = sys.argv[2]
    
    if not Path(img_path).exists():
        log(f"Immagine non trovata: {img_path}", "red")
        sys.exit(1)

    # 1. Carica Immagine Originale
    img = cv2.imread(img_path)
    
    # 2. Prepara il Filtro
    filters = []
    if filter_name == "clahe":
        filters.append(CLAHEFilter())
    elif filter_name == "grayscale":
        filters.append(GrayscaleFilter())
    elif filter_name == "structural":
        filters.append(StructuralEdgeFilter())
    elif filter_name == "bilateral":
        from cv.filters import BilateralFilter
        filters.append(BilateralFilter())
    elif filter_name == "bilateral_sharp":
        from cv.filters import BilateralSharpenFilter
        filters.append(BilateralSharpenFilter())
    else:
        log(f"Filtro {filter_name} non riconosciuto.", "red")
        sys.exit(1)
        
    pipeline = CVPipeline(filters)
    
    # 3. Applica il Filtro
    log(f"Applicando il filtro '{filter_name}' a {img_path}...", "cyan")
    processed_img = pipeline.process(img)
    
    # 4. Affianca le immagini (Prima e Dopo)
    comparison = np.hstack((img, processed_img))
    
    # 5. Salva il risultato
    output_path = "filter_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    log(f"Fatto! Risultato salvato in: {output_path}", "green")

if __name__ == "__main__":
    main()
