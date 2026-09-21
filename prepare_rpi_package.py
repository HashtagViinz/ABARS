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
        for d in (deploy_dir / "models").iterdir():
            if d.is_dir() and d.name.endswith("_ncnn"):
                d.rename(d.with_name(d.name + "_model"))
        print(f"✅ Modelli copiati in {deploy_dir}/models")
    else:
        print("❌ Cartella ncnn_models_ready non trovata. Sicuro di averli estratti?")
        
    # 1b. Copia la cartella cv/ con i filtri
    src_cv = Path("cv")
    if src_cv.exists():
        shutil.copytree(src_cv, deploy_dir / "cv")
        print(f"✅ Filtri CV copiati in {deploy_dir}/cv")
        
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
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

from cv.filters import BilateralFilter, BilateralSharpenFilter, DehazeFilter, CVPipeline

CLUSTER_METRICS = {
    "v26s_untuned_tiled_bilateral_d11_ncnn": {"Precision": 0.7675, "Recall": 0.5565, "mAP50": 0.5788, "F1": 0.645},
    "v26s_untuned_tiled-3_ncnn": {"Precision": 0.6741, "Recall": 0.6153, "mAP50": 0.6038, "F1": 0.643},
    "v26s_untuned_dehaze_bilateral_sharp_ncnn": {"Precision": 0.7624, "Recall": 0.5288, "mAP50": 0.5447, "F1": 0.624},
    "v26m_untuned_tiled_bilateral_sharp_ncnn": {"Precision": 0.7476, "Recall": 0.5623, "mAP50": 0.5741, "F1": 0.641},
    "v26m_untuned_tiled-6_ncnn": {"Precision": 0.7407, "Recall": 0.5643, "mAP50": 0.6069, "F1": 0.640},
    "v26n_untuned_tiled_bilateral_d11_sharp_ncnn": {"Precision": 0.7645, "Recall": 0.5223, "mAP50": 0.5653, "F1": 0.620},
    "v26n_untuned_tiled-2_ncnn": {"Precision": 0.7005, "Recall": 0.5560, "mAP50": 0.5766, "F1": 0.619}
}

def get_pipeline_for_model(model_name):
    filters = []
    if "dehaze" in model_name:
        filters.append(DehazeFilter())
    if "bilateral_d11_sharp" in model_name:
        filters.append(BilateralSharpenFilter(d=11, sigmaColor=85, sigmaSpace=85))
    elif "bilateral_d11" in model_name:
        filters.append(BilateralFilter(d=11, sigmaColor=85, sigmaSpace=85))
    elif "bilateral_sharp" in model_name:
        filters.append(BilateralSharpenFilter(d=11, sigmaColor=85, sigmaSpace=85))
    return CVPipeline(filters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models', help='Path to models dir')
    parser.add_argument('--source', default='test_images', help='Path to images dir or video file')
    args = parser.parse_args()
    
    models_dir = Path(args.models)
    source_path = Path(args.source)
    output_base = Path("benchmark_results")
    
    is_video = source_path.is_file() and source_path.suffix in ['.mp4', '.avi']
    if not is_video:
        images = list(source_path.glob('*.jpg'))
        print(f"Trovate {len(images)} immagini di test.")
        
    csv_file = open('benchmark_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model', 'Precision', 'Recall', 'mAP50', 'F1_Score', 'CV_Preprocess_ms', 'NCNN_Inference_ms', 'Total_Latency_ms', 'FPS'])
        
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        m_name = model_dir.name
        print(f"\\n{'='*50}\\nAvvio test per: {m_name}")
        
        # Identifica se _model suffix e pulisce per logica CV
        clean_name = m_name.replace('_model', '')
        pipeline = get_pipeline_for_model(clean_name)
        
        # Preprocessing OpenCV su disco (per non sfalsare Ultralytics e salvare i nomi corretti)
        temp_dir = Path("temp_filtered")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        cv_time_total = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            t0 = time.time()
            if pipeline.filters:
                img = pipeline.process(img)
            cv_time_total += (time.time() - t0)
            cv2.imwrite(str(temp_dir / img_path.name), img)
            
        avg_cv_ms = (cv_time_total / len(images)) * 1000 if images else 0
        
        try:
            model = YOLO(str(model_dir), task='detect')
        except Exception as e:
            print(f"Errore caricamento {m_name}: {e}")
            continue
            
        results = model.predict(
            source=str(temp_dir), 
            save=True, 
            project=str(output_base), 
            name=m_name, 
            exist_ok=True,
            verbose=False
        )
        
        if hasattr(results[0], 'speed'):
            avg_inference_ms = sum([r.speed['inference'] for r in results]) / len(results)
            total_latency_ms = avg_cv_ms + avg_inference_ms
            fps = 1000 / total_latency_ms
            
            print(f"\\n--- RISULTATI {m_name} ---")
            print(f"Latenza OpenCV (Filtri): {avg_cv_ms:.2f} ms")
            print(f"Latenza Rete (NCNN): {avg_inference_ms:.2f} ms")
            print(f"Latenza Totale (Reale): {total_latency_ms:.2f} ms")
            print(f"FPS Reali Combinati: {fps:.2f} fps")
            
            metrics = CLUSTER_METRICS.get(clean_name, {"Precision": 0, "Recall": 0, "mAP50": 0, "F1": 0})
            csv_writer.writerow([
                m_name, 
                metrics['Precision'], 
                metrics['Recall'], 
                metrics['mAP50'], 
                metrics['F1'], 
                round(avg_cv_ms, 2),
                round(avg_inference_ms, 2),
                round(total_latency_ms, 2), 
                round(fps, 2)
            ])
            csv_file.flush()
            
    csv_file.close()
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    print("\\n✅ Benchmark completato! Dati salvati in benchmark_metrics.csv")
            
if __name__ == "__main__":
    main()
""")
    print(f"✅ Script rpi_benchmark.py generato in {deploy_dir}")

    # 4. Genera Dockerfile per il Raspberry Pi
    dockerfile = deploy_dir / "Dockerfile"
    with open(dockerfile, "w") as f:
        f.write("""FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
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
    print(f"\\nTUTTO PRONTO! Ora puoi zippare la cartella.")

if __name__ == "__main__":
    main()
