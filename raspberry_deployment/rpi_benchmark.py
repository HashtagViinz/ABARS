import os
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

def run_sahi_inference(model, img, pipeline, overlap_ratio=0.1):
    img_h, img_w = img.shape[:2]
    overlap_x = int(img_w * overlap_ratio)
    overlap_y = int(img_h * overlap_ratio)
    tiles = [
        (0, 0, img_w // 2 + overlap_x, img_h // 2 + overlap_y),
        (img_w // 2 - overlap_x, 0, img_w, img_h // 2 + overlap_y),
        (0, img_h // 2 - overlap_y, img_w // 2 + overlap_x, img_h),
        (img_w // 2 - overlap_x, img_h // 2 - overlap_y, img_w, img_h)
    ]
    
    all_boxes, all_scores, all_classes = [], [], []
    cv_time_total, inference_time_total = 0.0, 0.0
    
    for (tx_min, ty_min, tx_max, ty_max) in tiles:
        tile_img = img[ty_min:ty_max, tx_min:tx_max].copy()
        
        t0 = time.time()
        if pipeline.filters:
            tile_img = pipeline.process(tile_img)
        cv_time_total += (time.time() - t0)
        
        results = model.predict(source=tile_img, save=False, verbose=False)
        if hasattr(results[0], 'speed'):
            inference_time_total += results[0].speed['inference']
            
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                abs_x1 = x1 + tx_min
                abs_y1 = y1 + ty_min
                abs_x2 = x2 + tx_min
                abs_y2 = y2 + ty_min
                all_boxes.append([float(abs_x1), float(abs_y1), float(abs_x2 - abs_x1), float(abs_y2 - abs_y1)])
                all_scores.append(float(conf[i]))
                all_classes.append(int(cls[i]))
                
    final_boxes, final_scores, final_classes = [], [], []
    if len(all_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, 0.25, 0.45)
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(all_boxes[i])
                final_scores.append(all_scores[i])
                final_classes.append(all_classes[i])
                
    return final_boxes, final_scores, final_classes, cv_time_total, inference_time_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models', help='Path to models dir')
    parser.add_argument('--source', default='test_images', help='Path to images dir')
    args = parser.parse_args()
    
    models_dir = Path(args.models)
    source_path = Path(args.source)
    output_base = Path("benchmark_results")
    os.makedirs(output_base, exist_ok=True)
    
    images = list(source_path.glob('*.jpg'))
    print(f"Trovate {len(images)} immagini di test (RAW giganti da analizzare in SAHI).")
        
    csv_file = open('benchmark_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model', 'Precision', 'Recall', 'mAP50', 'F1_Score', 'CV_Preprocess_ms_SAHI', 'NCNN_Inference_ms_SAHI', 'Total_Latency_ms', 'FPS_SAHI'])
        
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        m_name = model_dir.name
        print(f"\n{'='*50}\nAvvio test SAHI per: {m_name}")
        
        clean_name = m_name.replace('_model', '')
        pipeline = get_pipeline_for_model(clean_name)
        
        model_out_dir = output_base / m_name
        os.makedirs(model_out_dir, exist_ok=True)
        
        try:
            model = YOLO(str(model_dir), task='detect')
        except Exception as e:
            print(f"Errore caricamento {m_name}: {e}")
            continue
            
        total_cv_ms, total_inf_ms, total_sahi_ms = 0.0, 0.0, 0.0
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            t_sahi_start = time.time()
            
            # SAHI Inference
            boxes, scores, classes, cv_time, inf_time = run_sahi_inference(model, img, pipeline)
            
            # Draw and save
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                c = classes[i]
                s = scores[i]
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
                cv2.putText(img, f"cls{c} {s:.2f}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(str(model_out_dir / img_path.name), img)
            
            # Accumula i tempi (cv_time è in secondi, inf_time in ms)
            total_cv_ms += cv_time * 1000
            total_inf_ms += inf_time
            total_sahi_ms += (time.time() - t_sahi_start) * 1000
            
        avg_cv_ms = total_cv_ms / len(images) if images else 0
        avg_inf_ms = total_inf_ms / len(images) if images else 0
        avg_total_sahi = total_sahi_ms / len(images) if images else 0
        fps = 1000 / avg_total_sahi if avg_total_sahi > 0 else 0
        
        print(f"\n--- RISULTATI {m_name} ---")
        print(f"Latenza OpenCV (Filtri su 4 tiles): {avg_cv_ms:.2f} ms")
        print(f"Latenza Rete (NCNN su 4 tiles): {avg_inf_ms:.2f} ms")
        print(f"Latenza SAHI Completa (Reale per immagine 4K): {avg_total_sahi:.2f} ms")
        print(f"FPS Drone (SAHI): {fps:.2f} fps")
        
        metrics = CLUSTER_METRICS.get(clean_name, {"Precision": 0, "Recall": 0, "mAP50": 0, "F1": 0})
        csv_writer.writerow([
            m_name, 
            metrics['Precision'], 
            metrics['Recall'], 
            metrics['mAP50'], 
            metrics['F1'], 
            round(avg_cv_ms, 2),
            round(avg_inf_ms, 2),
            round(avg_total_sahi, 2), 
            round(fps, 2)
        ])
        csv_file.flush()
            
    csv_file.close()
    print("\n✅ Benchmark SAHI completato! Dati salvati in benchmark_metrics.csv")
            
if __name__ == "__main__":
    main()
