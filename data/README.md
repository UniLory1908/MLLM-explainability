# Dataset locale

Questa cartella non contiene il dataset COCO completo per evitare di appesantire la repo.

Per eseguire notebook e script servono almeno:

```text
data/
  annotations/
    instances_val2017.json
  val2017/
    *.jpg
```

Nel progetto locale originale erano presenti anche altri file COCO (`captions_*`, `person_keypoints_*`, ecc.), ma per gli script principali qui versionati il file indispensabile e' `instances_val2017.json` insieme alle immagini in `val2017/`.
