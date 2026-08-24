#!/bin/bash
echo "========================================="
echo "Avvio della Dashboard di MLflow..."
echo "========================================="
echo ""
echo "Quando vedi che il server è partito, apri il browser all'indirizzo:"
echo "👉  http://127.0.0.1:5000"
echo ""
echo "Premi CTRL+C per fermare il server."
echo "========================================="

# Controllo se siamo sul cluster in base a dove ci troviamo o a un argomento
if [ "$1" == "cluster" ]; then
    DB_PATH="sqlite:////home/vsalvatore/ABARS/mlflow.db"
    echo "[INFO] Modalità CLUSTER attivata (DB: $DB_PATH)"
else
    DB_PATH="sqlite:///mlflow.db"
    echo "[INFO] Modalità LOCALE attivata (DB: $DB_PATH)"
fi

# Esegue MLflow UI tramite uv
uv run mlflow ui --backend-store-uri $DB_PATH --host 127.0.0.1
