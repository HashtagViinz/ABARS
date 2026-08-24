# ABARS Project Conventions

Quando lavori su questo progetto, devi rispettare rigorosamente le seguenti regole:

1. **Esecuzione Comandi**: Usa sempre `uv run main.py <comando>` per eseguire gli script del progetto. Questo garantisce l'uso corretto dell'ambiente virtuale.
2. **Modelli Base YOLO**: I pesi pre-addestrati di base (es. `yolov8s.pt`) sono salvati in `YOLO/base_models/`. Il codice è già configurato tramite `ultralytics.settings` per scaricarli e leggerli da lì. Non spostarli e non metterli nella root del progetto.
3. **Modelli Addestrati**: I risultati dei tuoi addestramenti vengono salvati in `YOLO/trained_model/`.
4. **Tracking con MLflow**: 
   - MLflow è gestito nativamente dalle callback di Ultralytics. **NON** wrappare mai l'addestramento in blocchi `with mlflow.start_run():`.
   - Il tracking URI è dinamico: se la variabile d'ambiente `MLFLOW_TRACKING_URI` esiste (come sul cluster), viene usata quella. Altrimenti, il codice fa fallback in automatico su `sqlite:///mlflow.db` situato nella root del progetto.
5. **Hyperparameter Tuning**: I risultati del tuning si trovano in `YOLO/tuning_results/`. Il comando `train` carica automaticamente i migliori parametri da lì, a meno che non venga passato esplicitamente il flag `--baseline`.
