import os
import csv
from pathlib import Path
from ultralytics import YOLO

# Lista dei 7 migliori modelli finali (con NCNN e run id rimossi per mappare la cartella)
TOP_MODELS = [
    "v26m_untuned_tiled-6",
    "v26s_untuned_tiled-3",
    "v26s_untuned_tiled_bilateral_d11",
    "v26n_untuned_tiled-2",
    "v26m_untuned_tiled_bilateral_sharp",
    "v26n_untuned_tiled_bilateral_d11_sharp",
    "v26s_untuned_dehaze_bilateral_sharp"
]

def get_yaml_for_model(model_name):
    """
    Deduce il path del file YAML del dataset in base al nome del modello.
    Se il nome ha un run_id finale (es. -6), lo rimuove per trovare il nome del dataset.
    """
    # Rimuove l'identificativo della run alla fine (es: "-6", "-3") se presente
    base_name = model_name
    if "-" in model_name:
        base_name = model_name.rsplit("-", 1)[0]
    
    # Estrae i filtri: "v26m_untuned_tiled_bilateral_sharp" -> "tiled_bilateral_sharp"
    # Assumiamo che tutti contengano "_untuned_"
    if "_untuned_" in base_name:
        suffix = base_name.split("_untuned_")[1]
    else:
        suffix = "tiled"
        
    yaml_dir = f"dataset_{suffix}"
    yaml_file = f"uavod10_{suffix}.yaml"
    
    return os.path.join(yaml_dir, yaml_file)

def main():
    print("="*60)
    print("🚀 AVVIO VALUTAZIONE CLASSI 0 E 5 SUI MIGLIORI 7 MODELLI")
    print("="*60)
    
    output_csv = "metrics_classes_0_5.csv"
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "mAP50", "mAP50-95", "Precision", "Recall"])
        
        for model_name in TOP_MODELS:
            print(f"\n[{model_name}] Valutazione in corso...")
            
            # Cerca il file pesi best.pt nella cartella addestrata
            # (Adattare il path se sul cluster è diverso, es. runs/detect/...)
            pt_path = f"YOLO/trained_model/{model_name}/weights/best.pt"
            
            if not os.path.exists(pt_path):
                print(f"❌ ATTENZIONE: File pesi non trovato: {pt_path}")
                print("Assicurati che i modelli si trovino nel percorso corretto sul cluster.")
                continue
                
            yaml_path = get_yaml_for_model(model_name)
            if not os.path.exists(yaml_path):
                print(f"❌ ATTENZIONE: File YAML non trovato: {yaml_path}")
                continue
                
            try:
                # Carica il modello
                model = YOLO(pt_path)
                
                # LA MAGIA: classes=[0, 5] valuta solo Building e Prefabricated-house
                results = model.val(data=yaml_path, classes=[0, 5], verbose=False)
                
                # Estrae le metriche dal dizionario dei risultati
                map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
                map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
                precision = results.results_dict.get('metrics/precision(B)', 0.0)
                recall = results.results_dict.get('metrics/recall(B)', 0.0)
                
                print(f"✅ Completato! mAP50: {map50:.4f}, Precision: {precision:.4f}")
                
                writer.writerow([
                    model_name,
                    f"{map50:.4f}",
                    f"{map50_95:.4f}",
                    f"{precision:.4f}",
                    f"{recall:.4f}"
                ])
                f.flush()
                
            except Exception as e:
                print(f"❌ ERRORE durante la validazione di {model_name}: {e}")

    print("\n" + "="*60)
    print(f"🎉 Valutazione completata! Risultati salvati in: {output_csv}")
    print("="*60)

if __name__ == "__main__":
    main()
