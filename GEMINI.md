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
8. **Gestione Git e File Pesi**: 
   - I file dei pesi di base (`.pt`) NON devono mai essere tracciati da Git nella root del progetto (esiste la regola `/*.pt` nel `.gitignore` per evitarlo).
   - Qualsiasi peso di base o nuovo scaricamento deve avvenire esclusivamente dentro `YOLO/base_models/`.
9. **Valutazione Metriche**:
   - Se l'utente chiede un parere sui risultati di addestramento su MLflow, l'agente deve consultare le linee guida presenti in `YOLO/METRICS_GUIDE.md` (le Loss come `box_loss`, `cls_loss`, `dfl_loss` devono essere vicine allo 0, mentre `mAP` e `recall` vicine a 1).
10. **Pipeline di Computer Vision (Dataset Builder OOP)**:
    - Non applicare filtri di Computer Vision tramite "Monkey Patching" on-the-fly su YOLO/Albumentations. A causa dell'uso di `spawn` nei workers multiprocessing, YOLOv8 elude le modifiche runtime in memoria.
    - Tutti i filtri CV vanno scritti come oggetti puri (es. `CLAHEFilter`) dentro `cv/filters.py`.
    - Quando si lancia il train con il parametro `--filter` (es. `--filter clahe grayscale`), il codice usa `cv/dataset_builder.py` per generare fisicamente una copia temporanea su disco del dataset filtrato (es. `dataset_clahe_grayscale/`). Questo file yaml viene poi automaticamente passato a YOLO per un addestramento sicuro al 100%. Le cartelle generate sono coperte dal `.gitignore`.
11. **Sviluppo Immediatamente Successivo (Offline Tiling)**:
    - La Data Augmentation per variazioni (Mosaic, Flips, etc) è già coperta nativamente da YOLO.
    - Il prossimo step fondamentale da implementare è l'**Offline Tiling (Dataset Slicing)**: creare uno script che prende le immagini ad alta risoluzione, le taglia in piastrelle più piccole (es. grid 2x2) e ricalcola i Bounding Box. Questo moltiplicherà la dimensione del dataset ed eviterà che YOLO distrugga i piccoli oggetti (es. auto, edifici piccoli) quando comprime tutto a 640/1024px.
12. **Standard Script SLURM (Bash)**:
    - Qualsiasi script bash generato per il cluster DEVE tassativamente includere le risorse corrette (`#SBATCH --cpus-per-task=8`, `#SBATCH --gres=gpu:quadro_rtx_6000:2`).
    - DEVE esportare sempre l'intero blocco di sicurezza per l'ambiente e il multi-GPU:
      ```bash
      export MLFLOW_TRACKING_URI="sqlite:////home/vsalvatore/ABARS/mlflow.db" 
      export NCCL_P2P_DISABLE=1         # Disabilita il Peer-to-Peer diretto se la risorsa condivisa dà problemi
      export NCCL_IB_DISABLE=1          # Disabilita InfiniBand forzando i socket standard TCP
      export OMP_NUM_THREADS=1          # Evita conflitti di CPU nei worker
      export PYTHONUNBUFFERED=1         # Stampa i log in tempo reale senza buffering
      ```
    - Ove applicabile (nei training), deve includere le istruzioni commentate per l'uso dei filtri (es. `--filter clahe`).
13. **Parametri Deprecati in YOLOv8+**: Non passare **MAI** argomenti come `image_weights=True` o `fl_gamma` alla funzione `model.train()` (es. in `YOLO/model.py`). Ultralytics ha rimosso e blindato questi parametri nelle versioni v8/v11/v26. Inserirli causerà un immediato `SyntaxError`. Il bilanciamento delle classi per dataset fortemente sbilanciati viene gestito intrinsecamente dalla Loss function.
14. **YOLOv26 e Hyperparameter Tuning**: L'algoritmo genetico di tuning di Ultralytics può fallire catastroficamente con modelli NMS-free come YOLOv26 (es. impostando un `lr0=1e-05` che congela l'apprendimento per tutte le epoche). Per la famiglia v26, è fortemente raccomandato eseguire i training direttamente in modalità `--baseline` (untuned) ignorando il tuning.
15. **Gestione Errori SLURM (CUDA Out of Memory in DDP)**: Se un modello piccolo (es. Nano o Small) va in `CUDA out of memory` su SLURM poco dopo l'avvio (es. in `preprocess_batch`), il problema NON è il modello. La causa è quasi certamente un **processo Python zombie** rimasto incastrato nella VRAM (GPU) a seguito di un crash precedente (DDP escape). Dato che l'utente spesso non ha i permessi per killare processi zombie su nodi condivisi, la soluzione ufficiale è rilanciare il training escludendo esplicitamente il nodo infetto (es. `sbatch --exclude=node126 ...`).
