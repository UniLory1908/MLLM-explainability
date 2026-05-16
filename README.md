# MLLM Explainability

Repository condivisa per il progetto universitario su explainability nei Multimodal Large Language Models, con focus operativo su Qwen2-VL + TAM.

Questa repository contiene il codice e la documentazione minima per:
- preparare il materiale della fase 0;
- eseguire run locali con Qwen2-VL + TAM;
- lanciare il prompt sweep e le analisi successive;
- mantenere configurazioni e prompt set in file separati e leggibili.

Non vengono aggiunti qui i risultati della run notturna del prompt sweep. Gli output possono essere rigenerati in locale oppure condivisi separatamente quando servono.

## Struttura

- `scripts/runs/`: script eseguibili per i run.
- `scripts/analysis/`: script di analisi offline sugli artifact gia' prodotti.
- `scripts/common/`: utility condivise tra runner e analisi.
- `configs/`: configurazioni stabili, per esempio registro immagini.
- `prompt_sets/`: set di prompt e batch di immagini usati dagli sweep.
- `dataset_creation/`: notebook e file usati per costruire il subset COCO della fase 0.
- `docs/`: documentazione operativa, note e riferimenti.
- `external/tam-logit-lenses/`: codice TAM esterno richiesto dagli script.
- `data/`: cartella attesa per il dataset COCO locale, non versionata.

## Script principali

### Fase 0 e ispezione intermedia

- `scripts/runs/phase0_run_qwen_tam.py`
  Run locale della fase 0 su immagini selezionate.

- `scripts/runs/run_qwen_tam_intermediate.py`
  Salva viste intermedie utili per controllare heatmap e layer.

### Prompt sweep

- `scripts/runs/run_qwen_tam_prompt_sweep.py`
  Esegue uno sweep di prompt sulla stessa immagine e salva metadata, risposte, heatmap TAM per step e raw map TAM usate per lo scanpath.

- `scripts/runs/overnight_prompt_sweep.py`
  Lancia una batch run su piu' immagini e poi richiama in sequenza gli script di analisi.

- `scripts/runs/run_raw_scanpath_batch.ps1`
  Launcher locale per rigenerare i run scanpath basati su raw map TAM sulle immagini del batch.

### Analisi offline

- `scripts/analysis/analyze_prompt_sweep.py`
  Riepilogo base delle risposte generate.

- `scripts/analysis/analyze_prompt_sweep_v2.py`
  Prima misura di stabilita' delle heatmap.

- `scripts/analysis/analyze_prompt_sweep_v3.py`
  Versione piu' conservativa della v2.

- `scripts/analysis/analyze_prompt_sweep_v4.py`
  Matching controllato piu' flessibile della v3 ma piu' difendibile della v2.

- `scripts/analysis/analyze_misgrounding_v1.py`
  Ranking euristico di casi sospetti, da leggere come supporto qualitativo.

- `scripts/analysis/scanpath_viewer.py`
  Costruisce frame, GIF e contact sheet a partire dai metadata scanpath del prompt sweep. Se disponibili, usa le raw map TAM invece della JPG visuale.

- `scripts/analysis/validate_raw_scanpath_batch.py`
  Controlla che i run raw scanpath attesi abbiano metadata, raw map e hotspot dominanti coerenti.

- `scripts/analysis/summarize_overnight_runs.py`
  Sintesi compatta cross-image di una batch run gia' conclusa.

## Configurazioni utili

- `configs/image_registry.json`
  Registro delle immagini con ID COCO, alias leggibile e ruolo nel batch.

- `prompt_sets/prompt_sensitivity_v2.json`
  Prompt set usato per il prompt sweep.

- `prompt_sets/image_batch_overnight_v1.json`
  Batch di immagini usato per la run multi-image.

## Requisiti minimi

Installazione:

```bash
pip install -r requirements.txt
```

Gli script si aspettano anche il codice TAM presente in `external/tam-logit-lenses/ll_tam`.

## Dataset locale

Il dataset COCO completo non e' incluso nella repository. Va preparato in locale seguendo [data/README.md](data/README.md).

Struttura attesa:

```text
data/
  annotations/
    instances_val2017.json
  val2017/
    *.jpg
```

## Da dove iniziare

Per una vista rapida dei file e dell'ordine d'uso:
- `docs/prompt_sweep/COME_USARE_I_FILE.txt`
- `docs/prompt_sweep/Scanpath_Lettura.ipynb`

Per la parte di fase 0:
- `docs/phase0/Istruzioni fase 0.txt`

Per i riferimenti teorici:
- `docs/references/`

## Nota

La repository privilegia chiarezza e riproducibilita'. Per questo:
- gli script sono separati per fase;
- i prompt e le immagini stanno in file JSON dedicati;
- gli output pesanti non vengono aggiornati automaticamente nel mirror pubblico.
