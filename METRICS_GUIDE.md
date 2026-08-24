# Guida alla Lettura delle Metriche YOLO su MLflow

Questa guida ti aiuterà a interpretare tutti i valori che vedi nella dashboard di MLflow quando confronti due addestramenti di YOLO. Sapere cosa guardare ti permetterà di capire non solo *chi ha vinto*, ma *perché*.

---

## 📉 1. Le "Loss" (Gli Errori del Modello)
**Regola d'oro:** Per tutte le Loss, **più il valore è vicino a 0, MEGLIO È**. 
Le metriche che iniziano con `train/` indicano l'errore sui dati di addestramento, mentre quelle con `val/` indicano l'errore sui dati di validazione (quelli che il modello non ha mai visto). **I valori `val/` sono quelli che contano davvero per giudicare il modello.**

*   `val/box_loss` (Bounding Box Loss): 
    *   **Cosa significa:** Misura quanto i rettangoli (bounding box) disegnati dal modello siano distanti per dimensione e posizione da quelli reali. 
    *   **Se è basso:** Il modello disegna box precisissimi e aderenti all'oggetto.
*   `val/cls_loss` (Classification Loss):
    *   **Cosa significa:** Misura l'errore nel riconoscere la classe giusta (es. dire "Piscina" quando invece era "Nave").
    *   **Se è basso:** Il modello è molto sicuro di sé e sbaglia raramente classe.
*   `val/dfl_loss` (Distribution Focal Loss):
    *   **Cosa significa:** Misura "l'incertezza" sui bordi del box. Aiuta il modello a capire se il bordo di un edificio è netto o sfocato.
    *   **Se è basso:** Il modello riesce a trovare bordi definiti in modo eccellente.

---

## 📈 2. Le Metriche di Performance (I Risultati)
**Regola d'oro:** Per queste metriche, **più il valore è vicino a 1 (o al 100%), MEGLIO È**. 
La lettera "B" finale (es. `precisionB`) sta per "Boxes" (perché stiamo facendo object detection con i rettangoli).

*   `metrics/precisionB` (Precisione):
    *   **Cosa significa:** *Delle 100 cose che il modello ha detto essere "Edifici", quante lo erano davvero?* 
    *   **In breve:** Una precisione alta significa che il modello fa pochissimi **Falsi Positivi** (non prende abbagli).
*   `metrics/recallB` (Sensibilità / Richiamo):
    *   **Cosa significa:** *Dei 100 "Edifici" che c'erano effettivamente nella foto, quanti è riuscito a trovarne?*
    *   **In breve:** Una recall alta significa che il modello fa pochissimi **Falsi Negativi** (non si lascia sfuggire quasi nulla). Spesso, se alzi la recall perdi precisione, e viceversa.
*   `metrics/mAP50B` (Mean Average Precision al 50%):
    *   **Cosa significa:** È la metrica standard per l'Object Detection. Fa una media tra Precision e Recall, considerando "corretto" un box se si sovrappone per almeno il 50% a quello reale.
    *   **In breve:** È il voto principale del modello. Se è più alto, il modello in generale è migliore.
*   `metrics/mAP50-95B`:
    *   **Cosa significa:** Come la precedente, ma è severissima. Fa la media chiedendo al modello di sovrapporre i box al 50%, poi al 55%, fino al 95%. 
    *   **In breve:** Se questo valore è alto, il modello è un "cecchino" perfetto nel tracciare i box al millimetro.

---

## ⚙️ 3. Altri Parametri (Dietro le quinte)
Questi valori sono informativi e servono solo per capire come si sta muovendo l'algoritmo di addestramento:

*   `lr/pg0`, `lr/pg1`, `lr/pg2`: Sono i *Learning Rate* (tassi di apprendimento) per le diverse parti della rete neurale (Pesi, Bias, ecc.). Si abbasseranno da soli man mano che l'addestramento procede. Non si usano per giudicare il modello finito.
