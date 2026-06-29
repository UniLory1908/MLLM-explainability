# Run Inventory

Inventario operativo delle run locali. Serve per evitare confusione tra run
ufficiali, fix, baseline storiche e probe tecniche.

## Run Ufficiale V3

Pattern:

```text
outputs/prompt_sensitivity/<image>/v3_wordlevel_gpu_alllayers_<label>/
```

Immagini incluse:

```text
000000030213_kitchen_counter
000000331352_bathroom_toilet
000000426253_microwave_bottle
000000393226_street_traffic
000000555009_desk_monitor
```

Configurazione:

```text
prompt set: prompt_sets/prompt_sensitivity_v3_wordlevel.json
prompt: 8 per immagine
device: cuda
dtype: float16
max_new_tokens: 128
all_layers: true
layer effettivi: 29
```

Esito verificato:

```text
5/5 immagini completate
40/40 prompt completati
raw map presenti
metadata presenti
```

Dimensione indicativa:

```text
totale: circa 50.83 GB
kitchen_counter: circa 9.84 GB
bathroom_toilet: circa 14.00 GB
microwave_bottle: circa 6.65 GB
street_traffic: circa 9.91 GB
desk_monitor: circa 10.43 GB
```

Nota: quattro risposte della run a 128 token erano troncate. Per quei casi usare
la run fix256 descritta sotto.

## Fix 256 Token

Pattern:

```text
outputs/prompt_sensitivity/<image>/v3_wordlevel_gpu_alllayers_fix256_<label>/
```

Casi rilanciati:

```text
000000030213_kitchen_counter / baseline_neutral
000000393226_street_traffic / extra_knowledge_context
000000555009_desk_monitor / baseline_neutral
000000555009_desk_monitor / order_disruption_stress
```

Configurazione:

```text
device: cuda
dtype: float16
max_new_tokens: 256
all_layers: true
layer effettivi: 29
```

Esito verificato:

```text
kitchen_counter / baseline_neutral: 192 token, completo
street_traffic / extra_knowledge_context: 148 token, completo
desk_monitor / baseline_neutral: 200 token, completo
desk_monitor / order_disruption_stress: 176 token, completo
```

Regola d'uso:

```text
Quando un notebook o una tabella usa uno di questi quattro casi, preferire il
metadata della run fix256 al metadata della run principale a 128 token.
```

Dimensione indicativa:

```text
totale fix256: circa 16.37 GB
kitchen_counter: circa 4.43 GB
street_traffic: circa 3.51 GB
desk_monitor: circa 8.43 GB
```

## Baseline Storica V2f

Pattern:

```text
outputs/prompt_sensitivity/<image>/overnight_prompt_sensitivity_v2f_rawscanpath_<label>/
```

Uso:

```text
baseline storica
confronti metodologici
notebook word-level baseline gia' preparati
```

Non sostituisce la run V3 ufficiale.

## Probe Tecniche

Queste run servono come test tecnici, non come risultati finali:

```text
v3_wordlevel_gpu_alllayers_probe_20260515_bathroom_toilet
gpu_probe_alllayers_full32
gpu_probe_alllayers_small
gpu_probe_colleague_cot_lastlayer
```

La probe `v3_wordlevel_gpu_alllayers_probe_20260515_bathroom_toilet` pesa circa
14 GB. Ora che la run V3 completa e la fix256 esistono, non va usata nel report.

## Output Derivati

Cartelle leggere generate dai notebook:

```text
outputs/word_level_baseline_comparison/
outputs/word_level_all_prompts/
outputs/word_level_comparison/
```

Sono viste derivate, non dati primari. Si possono rigenerare dai metadata e dalle
raw map se necessario.

## Regole Di Pulizia

- Non cancellare la run V3 ufficiale.
- Non cancellare la fix256.
- Non cancellare la v2f rawscanpath finche' serve confronto storico.
- Prima candidata per liberare spazio: probe all-layers da 14 GB su bathroom.
- Non cancellare junction in `outputs/`.
- Non portare output pesanti nella repo condivisa senza richiesta esplicita.
