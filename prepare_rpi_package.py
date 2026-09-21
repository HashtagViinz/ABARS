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
import csv
from pathlib import Path
from ultralytics import YOLO

CLUSTER_METRICS = {
    "v26s_untuned_tiled_bilateral_d11_ncnn": {"Precision": 0.7675, "Recall": 0.5565, "mAP50": 0.5788, "F1": 0.645},
    "v26s_untuned_tiled-3_ncnn": {"Precision": 0.6741, "Recall": 0.6153, "mAP50": 0.6038, "F1": 0.643},
    "v26s_untuned_dehaze_bilateral_sharp_ncnn": {"Precision": 0.7624, "Recall": 0.5288, "mAP50": 0.5447, "F1": 0.624},
    "v26m_untuned_tiled_bilateral_sharp_ncnn": {"Precision": 0.7476, "Recall": 0.5623, "mAP50": 0.5741, "F1": 0.641},
    "v26m_untuned_tiled-6_ncnn": {"Precision": 0.7407, "Recall": 0.5643, "mAP50": 0.6069, "F1": 0.640},
    "v26n_untuned_tiled_bilateral_d11_sharp_ncnn": {"Precision": 0.7645, "Recall": 0.5223, "mAP50": 0.5653, "F1": 0.620},
    "v26n_untuned_tiled-2_ncnn": {"Precision": 0.7005, "Recall": 0.5560, "mAP50": 0.5766, "F1": 0.619}
}

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
        
    csv_file = open('benchmark_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model', 'Precision', 'Recall', 'mAP50', 'F1_Score', 'Latency_ms', 'FPS'])
        
    # Per ogni modello NCNN
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        m_name = model_dir.name
        print(f"\\n{'='*50}\\nAvvio test per: {m_name}")
        output_dir = output_base / m_name
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            model = YOLO(str(model_dir), task='detect')
        except Exception as e:
            print(f"Errore caricamento {m_name}: {e}")
            continue
            
        start_time = time.time()
        
        results = model.predict(
            source=str(source_path), 
            save=True, 
            project=str(output_base), 
            name=m_name, 
            exist_ok=True,
            verbose=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calcolo Latenza media
        if hasattr(results[0], 'speed'):
            avg_inference_ms = sum([r.speed['inference'] for r in results]) / len(results)
            fps = 1000/avg_inference_ms
            
            print(f"\\n--- RISULTATI {m_name} ---")
            print(f"Tempo Totale Esecuzione: {total_time:.2f} s")
            print(f"Latenza Media Rete (Inferenza Pura): {avg_inference_ms:.2f} ms")
            print(f"FPS Stimati (Puri): {fps:.2f} fps")
            
            # Scrittura nel CSV
            metrics = CLUSTER_METRICS.get(m_name, {"Precision": 0, "Recall": 0, "mAP50": 0, "F1": 0})
            csv_writer.writerow([
                m_name, 
                metrics['Precision'], 
                metrics['Recall'], 
                metrics['mAP50'], 
                metrics['F1'], 
                round(avg_inference_ms, 2), 
                round(fps, 2)
            ])
            csv_file.flush()
        else:
            print("Tempo totale misurato:", total_time)
            
    csv_file.close()
    print("\\n✅ Benchmark completato! Dati salvati in benchmark_metrics.csv")
            
if __name__ == "__main__":
    main()
""")
    print(f"✅ Script rpi_benchmark.py generato con successo (con esportazione CSV) in {deploy_dir}")

    # 4. Genera Dockerfile per il Raspberry Pi
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
