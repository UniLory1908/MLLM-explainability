# MLLM Explainability

Repository condivisa per il progetto universitario su explainability nei
Multimodal Large Language Models, con focus su Qwen2-VL e Token Attribution Maps
(TAM).

La repository contiene codice, documentazione e output leggeri. I raw output
pesanti, le mappe `.npy`, il database dashboard e i CSV completi delle metriche
sono condivisi separatamente quando necessario.

## Repository Structure

- `scripts/runs/`: runner per Qwen2-VL/TAM e prompt sweep.
- `scripts/analysis/`: analisi offline, visualizzazioni V3, query e analisi v6.
- `scripts/dashboard/`: dashboard Flask/SQLite per ispezione di casi, parole,
  layer, confronti e viste v6.
- `configs/`: configurazioni stabili, incluso il registro immagini.
- `prompt_sets/`: prompt set usati negli sweep.
- `docs/`: stato corrente, metriche, dashboard, archivio CSV e note operative.
- `outputs/analysis/`: output generati e ignorati, inclusa la v6 horizontal
  analysis quando rigenerata localmente.
- `dataset_creation/`: materiali fase 0.
- `ObjDetection/` e `CriscuoloCASA/`: materiali della linea Object
  Detection/Custom Dataset del collega.
- `external/tam-logit-lenses/`: codice TAM esterno richiesto dagli script.
- `data/`: cartella attesa per COCO locale, non completa in GitHub.

## Current TAM Snapshot

La snapshot TAM/statistical corrente copre:

- 100 immagini COCO;
- 8 prompt condition per immagine;
- 800 casi image x prompt;
- 995686 mappe indicizzate;
- 995683 righe di metriche per-map;
- 34334 layer scanpaths;
- 23200 word scanpaths.

Il focus e' il comportamento di attribuzione TAM-derived: concentrazione,
diffusione, centroidi, hotspot/regioni, multipeak structure, scanpath
word-wise/layer-wise e proxy diagnostici.

## Complete CSV Archive

I CSV completi delle metriche sono troppo grandi per essere versionati in GitHub
normale. Sono preparati per il caricamento in Drive condiviso:

Drive course folder: `FVAB 2025-2026 / Gruppo 17 / CSV completi/`

Il manifest dei file CSV, con righe, colonne, dimensioni, origine e comando di
rigenerazione, si trova in:

```text
docs/CSV_ARCHIVE_MANIFEST.md
```

La struttura locale generata dallo script di export e':

```text
outputs/statistical_archive/final_csv_archive_20260629/
```

L'export legge il DB dashboard e le metriche gia' calcolate. Non rilancia
inferenza Qwen/TAM.

## Final Report

Il report finale sul prompt ablation con TAM, dashboard e analisi orizzontale
v6 si trova in:

```text
docs/report/lorenzo/
```

`CriscuoloCASA/` e `ObjDetection/` contengono una linea separata su Object
Detection / Custom Dataset.

## Dashboard

Avvio locale:

```bash
python -m scripts.dashboard.app --host 127.0.0.1 --port 5050
```

URL locale:

```text
http://127.0.0.1:5050/
```

Route principali:

- `/`: indice casi;
- `/case/<case_id>`: singolo caso;
- `/case/<case_id>/matrix`: matrice word x layer;
- `/case/<case_id>/word/<word_index>`: dettaglio parola;
- `/compare`: confronto tra prompt sulla stessa immagine;
- `/analysis/v6`: viste v6 dataset-level;
- `/analysis/v6/findings`, `/prompts`, `/images`, `/bbox`,
  `/model-locations`, `/cases`, `/explorer`: viste di sintesi e ispezione.
- `/analysis/location-validation`: validazione supplementare dei 102 output
  spaziali parseable contro bounding box e maschere COCO.

Public dashboard:

```text
https://lorenzocastellano.net
```

Il dashboard richiede output locali pesanti non versionati:

```text
outputs/prompt_sensitivity/
outputs/dashboard_index/tam_index.sqlite
outputs/dashboard_cache/
```

## Environment

Installazione base:

```bash
pip install -r requirements.txt
```

Note operative e comandi principali:

```text
README_RUN.md
```

## Main Documentation

- `docs/CURRENT_STATE.md`
- `docs/DASHBOARD.md`
- `docs/DATA_AND_OUTPUTS.md`
- `docs/METRICS.md`
- `docs/V6_HORIZONTAL_ANALYSIS.md`
- `docs/CSV_ARCHIVE_MANIFEST.md`
- `outputs/RUN_INVENTORY.md`

## Commenti Finali

La pipeline TAM ha prodotto una base ispezionabile e quantitativa per
studiare come le TAM cambiano al variare del prompt. I risultati supportano
claim prudenti su prompt-associated changes in TAM-derived attribution behavior,
prompt fingerprints, concentrazione/diffusione, centroid shift, hotspot
structure, scanpath descriptors e separabilita' in metric space.

Questi risultati non dimostrano grounding corretto, localizzazione corretta o
causal faithfulness. Le scanpath sono traiettorie derivate dalle attribution
map, non eye-tracking umano. Gli output bbox/location-style sono trattati come
segnali di formato della risposta, non come prova che il modello abbia
localizzato correttamente l'oggetto.

La vista `/analysis/v6/model-locations` distingue i 139 output strict
bbox/location-style dai 102 casi con coordinate effettivamente parseable
(`order_disruption_stress`: 72, `colleague_obj_detection_hard`: 28,
`misleading_wrong_subject`: 2). Il report usa l'alias leggibile
`object_detection_hard` per la label dati `colleague_obj_detection_hard`.
La vista `/analysis/location-validation` valuta quei 102 output spaziali
parseable contro bounding box e maschere COCO. Riporta target matches, valid
alternative-object matches, casi ambigui e casi background/wrong. Questa
validazione riguarda le coordinate esplicite generate nella risposta, non la
causal faithfulness delle TAM.

Sviluppi futuri ragionevoli includono metriche perturbative piu' forti,
negative queries piu' bilanciate, analisi su piu' modelli e controlli
qualitativi mirati sui casi concordanti e discordanti.
