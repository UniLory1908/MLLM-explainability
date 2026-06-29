from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "runs" else SCRIPT_DIR
# Qui punto alla cartella del repository TAM che ho messo dentro il progetto.
# Mi serve aggiungerla al path per poter importare i file Python del tool.
LL_TAM_DIR = PROJECT_ROOT / "external" / "tam-logit-lenses" / "ll_tam"
sys.path.insert(0, str(LL_TAM_DIR))

# Funzioni prese dal file demo.py del repository TAM.
# - _build_logitlens_logits: prepara i logits intermedi da cui poi ricavo la TAM
# - _decode_tokens: converte gli id dei token in testo leggibile
# - _num_rounds: dice quante "round" multimodali ci sono nel prompt
from demo import _build_logitlens_logits, _decode_tokens, _num_rounds  # noqa: E402
# Funzione del repository TAM per calcolare le metriche finali tra heatmap e maschera GT.
from new_eval import compute_all_metrics  # noqa: E402
# Utility del repository Qwen/TAM che separa input immagine e testo nel formato atteso dal modello.
from qwen_utils import process_vision_info  # noqa: E402
# Classe principale del tool TAM: prende i logits del modello e costruisce la heatmap del token scelto.
from tam import TAM  # noqa: E402

# Prompt base del run finale.
# Qui non cerco una caption lunga, ma un nome oggetto semplice.
# Mi serve un output corto per trovare piu' facilmente un token target utile per TAM.
PROMPT_DEFAULT = "What is the main object in the image? Answer with one word only."
# Su questa immagine il prompt base tendeva a restituire "bathroom".
# "bathroom" pero' non e' un oggetto COCO che posso usare bene per le metriche,
# quindi qui rendo il prompt piu' specifico per far emergere "toilet".
PROMPT_OVERRIDES = {
    403385: "What bathroom object is most visible in the image? Answer with one word only.",
}
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def step_artifact_stem(token_label: str, step_idx: int) -> str:
    # Mantengo la stessa convenzione leggibile usata dal prompt sweep.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(token_label).strip().lower())
    safe = safe.strip("_")[:30] or "tok"
    return f"step_{step_idx:04d}_{safe}"


def load_binary_mask(mask_path: Path) -> np.ndarray:
    # Riapro una maschera binaria gia' salvata su disco.
    # Qui non la uso tanto, ma la tengo come utility utile per controlli veloci.
    return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)


def build_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    # Creo un overlay molto semplice: oggetto in rosso sopra l'immagine.
    # Serve piu' per controllo visivo che per i calcoli.
    image_np = np.array(image).copy()
    color_np = np.array((255, 0, 0), dtype=np.float32)
    image_np[mask.astype(bool)] = (
        (1 - alpha) * image_np[mask.astype(bool)] + alpha * color_np
    ).astype(np.uint8)
    return Image.fromarray(image_np)


def save_mask_and_overlay(
    image_path: Path,
    image: Image.Image,
    mask: np.ndarray,
    object_name: str,
    ann_id: int,
    masks_root: Path,
    overlays_root: Path,
) -> tuple[Path, Path]:
    # Qui risalvo maschera e overlay dell'oggetto che il modello ha effettivamente nominato.
    # Questo passaggio e' importante perche' il target finale puo' cambiare rispetto a quello iniziale.
    # Quindi voglio salvare la ground truth giusta del target davvero usato per le metriche.
    stem = image_path.stem
    obj_dir = masks_root / stem
    obj_dir.mkdir(parents=True, exist_ok=True)
    mask_path = obj_dir / f"{object_name}_{ann_id}.png"
    overlay_path = overlays_root / f"{stem}_{object_name}_{ann_id}_overlay.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    build_overlay(image, mask).save(overlay_path)
    return mask_path, overlay_path


def build_messages(image_path: str, prompt_text: str) -> list[dict]:
    # Costruisco il messaggio multimodale minimo per Qwen2-VL:
    # una immagine + un prompt testuale.
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]


def find_object_match(
    token_labels: list[str],
    output_text: str,
    dataset_row: pd.Series,
) -> tuple[str | None, int | None, int | None]:
    # Qui collego l'output del modello a uno degli oggetti annotati nel dataset.
    # Questo e' il passaggio che decide quale target uso davvero per TAM e metriche.
    #
    # Prima guardo obj_main/ann_id_main, cioe' il riferimento principale del dataset.
    # Poi guardo anche gli altri oggetti annotati, nel caso il modello scelga un oggetto secondario.
    objects: list[tuple[str, int, int]] = []
    for obj_col, ann_col in (
        ("obj_main", "ann_id_main"),
        ("obj_2", "ann_id_2"),
        ("obj_1", "ann_id_1"),
    ):
        obj_name = dataset_row.get(obj_col)
        ann_id = dataset_row.get(ann_col)
        if pd.notna(obj_name) and pd.notna(ann_id):
            key = (str(obj_name).lower(), int(ann_id))
            if key not in [(name, ann) for name, ann, _ in objects]:
                objects.append((str(obj_name).lower(), int(ann_id), len(objects) + 1))

    cleaned_tokens = [t.strip().lower() for t in token_labels]
    # Primo tentativo: match diretto sui token generati.
    for idx, tok in enumerate(cleaned_tokens):
        for obj_name, ann_id, obj_idx in objects:
            if tok == obj_name or obj_name in tok or tok in obj_name:
                return obj_name, idx, ann_id

    output_words = output_text.lower().replace(",", " ").split()
    # Secondo tentativo: match sulle parole dell'output completo.
    # Questo aiuta in casi tipo "polar bear", dove il token utile e' "bear".
    for word in output_words:
        for obj_name, ann_id, obj_idx in objects:
            if word == obj_name or obj_name in word or word in obj_name:
                for idx, tok in enumerate(cleaned_tokens):
                    if tok == word or word in tok or tok in word:
                        return obj_name, idx, ann_id

    return None, None, None


def main() -> None:
    # Qui parte il run finale vero e proprio.
    # L'idea e':
    # 1. leggere i file preparati dal notebook;
    # 2. caricare il modello una sola volta;
    # 3. fare il run sulle 5 immagini finali;
    # 4. aggiornare i CSV con output, token, heatmap e metriche.
    phase0_dir = PROJECT_ROOT / "outputs" / "phase0"
    annotation_file = PROJECT_ROOT / "data" / "annotations" / "instances_val2017.json"
    masks_root = phase0_dir / "masks"
    overlays_root = phase0_dir / "overlays"
    heatmaps_root = phase0_dir / "heatmaps"
    heatmaps_root.mkdir(parents=True, exist_ok=True)

    test_run_path = phase0_dir / "test_run_table.csv"
    dataset_path = phase0_dir / "dataset.csv"

    test_df = pd.read_csv(test_run_path)
    dataset_df = pd.read_csv(dataset_path)
    coco = COCO(str(annotation_file))

    # Alcune colonne devono restare testuali.
    # Se non lo faccio, pandas prova a trattarle come numeri e poi da' errore
    # quando provo a scriverci dentro stringhe come "bench" o "toilet".
    for col in [
        "prompt_finale",
        "target_token_atteso",
        "oggetto_principale",
        "nota",
        "mask_path",
        "overlay_path",
        "output_modello",
        "target_token_scelto",
        "heatmap_path",
    ]:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype("object")

    dataset_lookup = {int(row["img_id"]): row for _, row in dataset_df.iterrows()}

    # Carico modello e processor una sola volta.
    # Questa e' la parte pesante, quindi non va fatta dentro il loop.
    print("Loading model:", MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Questo blocco arriva dalla logica tecnica del repository TAM usato per Qwen2-VL.
    # In pratica dice a TAM dove stanno i token immagine, prompt e risposta.
    special_ids = {
        "img_id": [151652, 151653],
        "prompt_id": [151653, [151645, 198, 151644, 77091]],
        "answer_id": [[198, 151644, 77091, 198], -1],
    }

    results_rows = []
    final_mask_rows = []

    for row_idx, row in test_df.iterrows():
        # Lavoro una immagine alla volta.
        # Per ogni immagine voglio ottenere:
        # - output del modello
        # - target token davvero usato
        # - heatmap TAM del token
        # - metriche contro la maschera corretta
        img_id = int(row["img_id"])
        image_path = Path(row["path"])
        prompt_text = PROMPT_OVERRIDES.get(img_id, PROMPT_DEFAULT)
        dataset_row = dataset_lookup[img_id]

        print(f"\n[{img_id}] prompt: {prompt_text}")
        messages = build_messages(str(image_path), prompt_text)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Qui faccio la generazione vera del modello.
        # Chiedo anche gli hidden states perche' poi servono a TAM.
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=12,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        elapsed = round(time.time() - start, 2)

        generated_ids = outputs.sequences
        prompt_len = inputs["input_ids"].shape[1]
        token_labels = _decode_tokens(outputs, prompt_len, processor)
        num_rounds, _ = _num_rounds(outputs, prompt_len)
        output_text = processor.tokenizer.decode(
            generated_ids[0][prompt_len:].tolist(),
            skip_special_tokens=True,
        ).strip()

        # Qui scelgo l'oggetto/target da usare davvero per TAM e metriche.
        # Questo e' il collegamento tra output del modello e annotazioni COCO.
        object_name, step_idx, ann_id = find_object_match(token_labels[:num_rounds], output_text, dataset_row)
        print(f"[{img_id}] output: {output_text} | chosen object: {object_name} | step: {step_idx} | secs: {elapsed}")

        target_token = ""
        heatmap_path = ""
        obj_iou = np.nan
        func_iou = np.nan
        f1_iou = np.nan
        mask_path = row["mask_path"]
        overlay_path = row["overlay_path"]

        if object_name is not None and step_idx is not None and ann_id is not None:
            target_token = object_name
            # Se trovo un target valido, qui rigenero anche la maschera giusta dell'oggetto scelto.
            # Mi serve per essere sicuro che la metrica venga calcolata sul target corretto.
            image = Image.open(image_path).convert("RGB")
            ann = coco.loadAnns([ann_id])[0]
            mask = (coco.annToMask(ann) > 0).astype(np.uint8)
            mask_file, overlay_file = save_mask_and_overlay(
                image_path=image_path,
                image=image,
                mask=mask,
                object_name=object_name,
                ann_id=ann_id,
                masks_root=masks_root,
                overlays_root=overlays_root,
            )
            mask_path = str(mask_file)
            overlay_path = str(overlay_file)

            # Questa shape serve a TAM per ricostruire correttamente la parte visiva.
            # Senza questa informazione la heatmap non si riallinea bene all'immagine.
            vision_shape = (
                int(inputs["image_grid_thw"][0, 1]) // 2,
                int(inputs["image_grid_thw"][0, 2]) // 2,
            )
            logits = _build_logitlens_logits(
                outputs,
                model,
                len(outputs.hidden_states[0]) - 1,
                len(outputs.hidden_states[0]),
            )
            # Qui genero e salvo la heatmap TAM del token scelto davvero.
            heatmap_file = heatmaps_root / f"{img_id}_{step_artifact_stem(target_token, step_idx)}.jpg"
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape,
                logits,
                special_ids,
                image_inputs,
                processor,
                str(heatmap_file),
                int(step_idx),
                [],
                False,
            )
            # Qui calcolo le metriche finali della fase 0 contro la maschera COCO.
            # Quelle che mi interessano davvero per la consegna sono:
            # obj_iou, func_iou e f1_iou.
            metrics = compute_all_metrics(img_map, mask)
            heatmap_path = str(heatmap_file)
            obj_iou = float(metrics["obj_iou"])
            func_iou = float(metrics["func_iou"])
            f1_iou = float(metrics["f1_iou"])

            final_mask_rows.append({
                "img_id": img_id,
                "ann_id_principale": ann_id,
                "oggetto_principale": object_name,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
            })
        else:
            # Se non trovo un target valido, preferisco lasciare la riga vuota
            # piuttosto che assegnare metriche al target sbagliato.
            print(f"[{img_id}] no usable target token found in generated output")

        # Aggiorno la tabella finale riga per riga.
        # Questa tabella e' il riferimento unico del run completo.
        test_df.loc[row_idx, "prompt_finale"] = prompt_text
        test_df.loc[row_idx, "target_token_atteso"] = object_name or row["target_token_atteso"]
        test_df.loc[row_idx, "oggetto_principale"] = object_name or row["oggetto_principale"]
        test_df.loc[row_idx, "ann_id_principale"] = ann_id if ann_id is not None else row["ann_id_principale"]
        test_df.loc[row_idx, "mask_path"] = mask_path
        test_df.loc[row_idx, "overlay_path"] = overlay_path
        test_df.loc[row_idx, "output_modello"] = output_text
        test_df.loc[row_idx, "target_token_scelto"] = target_token
        test_df.loc[row_idx, "heatmap_path"] = heatmap_path
        test_df.loc[row_idx, "obj_iou"] = obj_iou
        test_df.loc[row_idx, "func_iou"] = func_iou
        test_df.loc[row_idx, "f1_iou"] = f1_iou

        results_rows.append({
            "img_id": img_id,
            "prompt": prompt_text,
            "output": output_text,
            "target_token": target_token,
            "obj_iou": obj_iou,
            "func_iou": func_iou,
            "f1_iou": f1_iou,
        })

    # Alla fine riscrivo tutti i CSV importanti gia' aggiornati con i risultati reali del run.
    # Cosi' notebook e script restano allineati sugli stessi file finali.
    final_masks_df = pd.DataFrame(final_mask_rows).drop_duplicates(subset=["img_id"], keep="last")
    final_masks_df.to_csv(phase0_dir / "final_test_masks.csv", index=False)
    test_df.to_csv(test_run_path, index=False)
    test_df.to_csv(phase0_dir / "final_test_worktable.csv", index=False)
    test_df[
        [
            "img_id",
            "path",
            "prompt_finale",
            "target_token_atteso",
            "ann_id_principale",
            "oggetto_principale",
            "nota",
        ]
    ].to_csv(phase0_dir / "final_test_prompts.csv", index=False)
    test_df[
        [
            "img_id",
            "path",
            "prompt_finale",
            "oggetto_principale",
            "ann_id_principale",
            "target_token_atteso",
        ]
    ].rename(columns={"prompt_finale": "prompt_suggerito"}).assign(
        note="Aggiornato dopo il run TAM."
    ).to_csv(phase0_dir / "selected_test_images.csv", index=False)

    # Questo e' il file finale piu' vicino alla consegna della fase 0.
    pd.DataFrame(results_rows).to_csv(phase0_dir / "results_phase0.csv", index=False)

    # Controllo finale minimo: 5 righe valide con target e metriche.
    valid_rows = sum(1 for row in results_rows if row["target_token"] and pd.notna(row["f1_iou"]))
    print(f"Valid rows with target token and metrics: {valid_rows}/{len(results_rows)}")
    print("\nSaved:", phase0_dir / "results_phase0.csv")


if __name__ == "__main__":
    main()
