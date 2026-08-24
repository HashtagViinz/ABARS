# ABARS Project Conventions

Quando lavori su questo progetto, devi rispettare rigorosamente le seguenti regole:

1. **Esecuzione Comandi**: Usa sempre `uv run main.py <comando>` per eseguire gli script del progetto. Questo garantisce l'uso corretto dell'ambiente virtuale.
2. **Modelli Base YOLO**: I pesi pre-addestrati di base (es. `yolov8s.pt`) sono salvati in `YOLO/base_models/`. Il codice è già configurato tramite `ultralytics.settings` per scaricarli e leggerli da lì. Non spostarli e non metterli nella root del progetto.
3. **Modelli Addestrati**: I risultati dei tuoi addestramenti vengono salvati in `YOLO/trained_model/`.
4. **Tracking con MLflow**: 
   - MLflow è gestito nativamente dalle callback di Ultralytics. **NON** wrappare mai l'addestramento in blocchi `with mlflow.start_run():`.
   - Il tracking URI è dinamico: se la variabile d'ambiente `MLFLOW_TRACKING_URI` esiste (come sul cluster), viene usata quella. Altrimenti, il codice fa fallback in automatico su `sqlite:///mlflow.db` situato nella root del progetto.
5. **Hyperparameter Tuning**: I risultati del tuning si trovano in `YOLO/tuning_results/`. Il comando `train` carica automaticamente i migliori parametri da lì, a meno che non venga passato esplicitamente il flag `--baseline`.
6. **Consultazione MLflow**: 
   - Usa lo script `./start_mlflow.sh` per lanciare rapidamente la dashboard (usa `./start_mlflow.sh cluster` se ti trovi sul nodo remoto).
   - Ultralytics salva i run su MLflow raggruppandoli sotto un Esperimento che ha come nome il **percorso assoluto di salvataggio** (es. `/home/vsalvatore/ABARS/YOLO/trained_model`). I singoli "Run" all'interno avranno il nome che passi tramite l'argomento `--name` da CLI.
7. **Strategia Modelli Aerei (UAV)**:
   - Il tuning genetico di YOLOv8 su questo dataset produce box nettamente più precisi (Loss molto più basse) anche se può apparire un leggero calo di Recall. 
   - La strategia ufficiale del progetto prevede di **usare sempre i parametri Tuned** e compensare i falsi negativi (Recall) implementando **pipeline di Computer Vision (CV)** in fase di pre-processing (es. filtri CLAHE, dehazing, sharpening) per esaltare i dettagli sfocati o in ombra.
