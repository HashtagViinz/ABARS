import os
import shutil
import random
from pathlib import Path

def main():
    deploy_dir = Path("raspberry_deployment")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    
    os.makedirs(deploy_dir)
    
    # 1. Copia i modelli NCNN
    src_models = Path("ncnn_models_ready")
    if src_models.exists():
        shutil.copytree(src_models, deploy_dir / "models")
        print(f"✅ Modelli copiati in {deploy_dir}/models")
    else:
        print("❌ Cartella ncnn_models_ready non trovata. Sicuro di averli estratti?")
        
    # 2. Prepara 100 immagini di test
    src_images = Path("dataset/images/test")
    target_images = deploy_dir / "test_images"
    os.makedirs(target_images, exist_ok=True)
    if src_images.exists():
        all_imgs = list(src_images.glob("*.jpg"))
        random.seed(42) # Riproducibile
        sample_imgs = random.sample(all_imgs, min(100, len(all_imgs)))
        for img in sample_imgs:
            shutil.copy(img, target_images / img.name)
        print(f"✅ {len(sample_imgs)} immagini copiate in {deploy_dir}/test_images")
        
    # 3. Genera lo script di benchmark (rpi_benchmark.py)
    benchmark_script = deploy_dir / "rpi_benchmark.py"
    with open(benchmark_script, "w") as f:
        f.write("""import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models', help='Path to models dir')
    parser.add_argument('--source', default='test_images', help='Path to images dir or video file')
    args = parser.parse_args()
    
    models_dir = Path(args.models)
    source_path = Path(args.source)
    output_base = Path("benchmark_results")
    
    # Check if source is image dir or video
    is_video = source_path.is_file() and source_path.suffix in ['.mp4', '.avi']
    if not is_video:
        images = list(source_path.glob('*.jpg'))
        print(f"Trovate {len(images)} immagini di test.")
    else:
        print(f"Test su video {source_path.name}")
        
    # Per ogni modello NCNN
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        print(f"\\n{'='*50}\\nAvvio test per: {model_dir.name}")
        output_dir = output_base / model_dir.name
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            model = YOLO(str(model_dir), task='detect')
        except Exception as e:
            print(f"Errore caricamento {model_dir.name}: {e}")
            continue
            
        start_time = time.time()
        
        # Inferenza
        # YOLO handle save=True automatically
        results = model.predict(
            source=str(source_path), 
            save=True, 
            project=str(output_base), 
            name=model_dir.name, 
            exist_ok=True,
            verbose=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calcolo Latenza media dai profili di Ultralytics
        if hasattr(results[0], 'speed'):
            # results[0].speed è un dict con preprocess, inference, postprocess (ms)
            avg_inference_ms = sum([r.speed['inference'] for r in results]) / len(results)
            print(f"\\n--- RISULTATI {model_dir.name} ---")
            print(f"Tempo Totale Esecuzione: {total_time:.2f} s")
            print(f"Latenza Media Rete (Inferenza Pura): {avg_inference_ms:.2f} ms")
            print(f"FPS Stimati (Puri): {1000/avg_inference_ms:.2f} fps")
            print(f"Immagini salvate in: {output_dir}")
        else:
            print("Tempo totale misurato:", total_time)
            
if __name__ == "__main__":
    main()
""")
    print(f"✅ Script rpi_benchmark.py generato in {deploy_dir}")

    # 4. Genera Dockerfile per il Raspberry Pi (basato su ARM64/Python)
    dockerfile = deploy_dir / "Dockerfile"
    with open(dockerfile, "w") as f:
        f.write("""FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "rpi_benchmark.py"]
""")
    print(f"✅ Dockerfile generato in {deploy_dir}")

    # 5. Genera requirements.txt
    req_file = deploy_dir / "requirements.txt"
    with open(req_file, "w") as f:
        f.write("""ultralytics>=8.0.0
opencv-python-headless
ncnn
""")
    print(f"✅ requirements.txt generato in {deploy_dir}")
    
    print(f"\\nTUTTO PRONTO! Ora puoi zippare la cartella con: zip -r raspberry_deployment.zip raspberry_deployment")

if __name__ == "__main__":
    main()
