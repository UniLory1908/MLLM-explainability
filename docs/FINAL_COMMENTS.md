# Commenti Finali

La pipeline TAM ha trasformato i raw output TAM in un flusso ispezionabile e
quantitativo: dashboard locale, archivio statistico, tabelle CSV/Parquet,
metriche per-map, regioni/hotspot, scanpath TAM-derived e analisi v6
dataset-level su 100 immagini x 8 prompt.

Le metriche mostrano differenze misurabili nel comportamento di attribuzione
TAM-derived tra prompt condition. In particolare, permettono di descrivere
prompt-associated changes in concentration, diffusion, centroid shift, hotspot
structure, multipeak behavior, scanpath descriptors, bbox/location-style output
signals and prompt metric fingerprints.

Questi risultati vanno interpretati come analisi diagnostica ed esplorativa.
Non dimostrano che il modello abbia un grounding corretto, non misurano causal
faithfulness e non costituiscono una valutazione ground-truth della
localizzazione. Le scanpath sono derivate dalle mappe di attribuzione TAM e non
sono eye-tracking umano. Gli output bbox/location-style sono segnali del formato
della risposta testuale e non equivalgono a localizzazione corretta.

I prossimi passi piu' solidi sono:

- confrontare le TAM con annotazioni spaziali COCO o maschere/bounding box;
- aggiungere test perturbativi o deletion/insertion per causal faithfulness;
- bilanciare meglio negative queries e prompt stress conditions;
- estendere l'analisi a piu' modelli o piu' prompt family;
- mantenere una revisione qualitativa dei casi concordanti e discordanti
  selezionati dalle metriche.
