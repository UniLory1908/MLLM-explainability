# MLLM Explainability

Repository condivisa per il progetto universitario su explainability nei Multimodal Large Language Models, con focus operativo su Qwen2-VL + TAM e sul materiale preparatorio della fase 0.

## Struttura

- `phase0_run_qwen_tam.py`: script principale per eseguire Qwen2-VL + TAM sul subset selezionato e calcolare le metriche finali.
- `run_qwen_tam_intermediate.py`: script di analisi intermedia per salvare heatmap e viste layer-by-layer.
- `dataset_creation/`: notebook e artefatti usati per costruire il subset COCO e i file CSV di fase 0.
- `outputs/`: risultati gia' prodotti con gli script.
- `docs/`: documenti di progetto, istruzioni e materiale bibliografico utile.
- `external/tam-logit-lenses/`: codice TAM esterno usato dagli script.
- `data/`: cartella attesa per dataset COCO locale, non versionata.

## Workflow attuale

1. Preparazione del subset COCO e delle maschere con il notebook in `dataset_creation/`.
2. Esecuzione dello script `phase0_run_qwen_tam.py` per il run finale su immagini selezionate.
3. Analisi piu' dettagliata con `run_qwen_tam_intermediate.py` quando serve ispezionare i layer.

## Requisiti

Installazione minima:

```bash
pip install -r requirements.txt
```

Gli script si aspettano anche il codice TAM presente in `external/tam-logit-lenses/ll_tam`, gia' incluso in questa repo.

## Dataset locale

Il dataset COCO completo non e' incluso nella repository. Va ricreato in locale seguendo [data/README.md](data/README.md).

Struttura attesa:

```text
data/
  annotations/
    instances_val2017.json
  val2017/
    *.jpg
```

## Documentazione utile

- `docs/phase0/`: istruzioni operative della fase 0.
- `docs/references/`: paper e materiale di riferimento su TAM ed explainability.
- `docs/prompt_ablation/`: materiale preliminare per la parte di prompt ablation.

## Note di collaborazione

- versionare codice, notebook, CSV e output piccoli/riproducibili;
- non versionare dataset COCO completo, ambienti virtuali, cache e output temporanei molto pesanti;
- mantenere in root solo gli script Python di utilizzo effettivo di TAM.
