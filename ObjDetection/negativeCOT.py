import os
import argparse 
import random
import torch
import pandas as pd
import numpy as np
import cv2
import gc
import shutil
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_utils import process_vision_info
from pycocotools.coco import COCO
from tam import TAM

# --- CONFIGURAZIONE ARGOMENTI DA RIGA DI COMANDO (CLI) ---
parser = argparse.ArgumentParser(description="Esegui test CoT su Negative Queries (Allucinazioni).")
parser.add_argument(
    '-n', '--num_images', 
    type=int, 
    default=None, 
    help="Numero massimo di immagini da analizzare (es. -n 5). Se omesso, analizza l'intero CSV."
)
args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_negative_cot/'

tam_dir = os.path.join(output_dir, 'tam_heatmaps_hallucinations_cot')
os.makedirs(tam_dir, exist_ok=True)

results_file = os.path.join(output_dir, 'results_negative_queries_cot.csv') 

print("Inizializzo COCO e le categorie...")
coco = COCO(ann_file)
df = pd.read_csv(csv_path)

# Recuperiamo tutte le categorie di COCO
cats = coco.loadCats(coco.getCatIds())
tutte_le_categorie = {cat['id']: cat['name'] for cat in cats}

if args.num_images is not None:
    df = df.head(args.num_images)
    print(f"\n[INFO] 🛠️ Modalità Test CLI: Elaborazione limitata alle prime {args.num_images} immagini!\n")

print("Carico il modello Qwen2-VL (versione CPU)...")
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=True
)

special_ids = {'img_id': [151652, 151653],
               'prompt_id': [151653, [151645, 198, 151644, 77091]],
               'answer_id': [[198, 151644, 77091, 198], -1]}

risultati = []

# --- VARIABILI PER METRICHE DI SPECIFICITY ---
TN = 0  # True Negative (Ragiona e conclude No)
FP = 0  # False Positive (Allucinazione: si autoconvince che ci sia)

# --- CICLO SULLE IMMAGINI ---
for index, row in df.iterrows():
    img_id = row['img_id']
    img_path = os.path.join(output_dir, row['path'])
    
    # 1. Troviamo quali oggetti sono REALMENTE nell'immagine
    ann_ids_immagine = coco.getAnnIds(imgIds=img_id)
    anns_immagine = coco.loadAnns(ann_ids_immagine)
    id_categorie_presenti = set([ann['category_id'] for ann in anns_immagine])
    
    # 2. Creiamo una lista di oggetti sicuramente ASSENTI e ne peschiamo uno
    id_categorie_assenti = [cat_id for cat_id in tutte_le_categorie.keys() if cat_id not in id_categorie_presenti]
    fake_obj_id = random.choice(id_categorie_assenti)
    fake_obj = tutte_le_categorie[fake_obj_id]
    
    print(f"\n[{index+1}/{len(df)}] Analizzo: {row['path']} | Target Falso (Assente): {fake_obj}")
    
    # 3. Costruzione del PROMPT COT
    prompt = f"Is there a {fake_obj} in this image? Let's think step by step before answering."
    system_prompt = "You are an AI detective. You must observe the image in detail, list what you see, and logically deduce whether the specified object is present."

    messages = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": img_path}, 
                {"type": "text", "text": prompt}
            ]
        }
    ]    
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=True,
            temperature=0.7,  # Un po' di creatività nel ragionamento
            use_cache=True, 
            output_hidden_states=True, 
            return_dict_in_generate=True
        )
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"   -> Ragionamento del modello:\n      {testo_generato}\n")
    
    # 4. VALUTAZIONE LOGICA DEL COT
    testo_pulito = testo_generato.lower()
    parole_generate = [ "".join(c for c in p if c.isalpha()) for p in testo_pulito.split() ]
    
    ha_detto_yes = "yes" in parole_generate
    ha_detto_no = "no" in parole_generate
    
    # Se conclude con un NO chiaro senza YES, ha resistito
    if ha_detto_no and not ha_detto_yes:
        print("   [+] Vero Negativo (TN): Il modello ha ragionato e resistito all'allucinazione.")
        TN += 1
        
        risultati.append({
            'img_id': img_id, 'fake_obj': fake_obj, 'prompt': prompt, 'output': testo_generato.replace('\n', ' '), 
            'target_token': 'NONE', 'token_index': -1, 'obj_iou': 0.0, 'is_hallucination': 0, 'is_correct': 1
        })
        
        del inputs, image_inputs, video_inputs, outputs, generated_ids
        gc.collect()
        
    else:
        # Altrimenti, si è confuso o ha allucinato (False Positive)
        print("   [!] ALLUCINAZIONE (FP): Il modello si è autoconvinto della presenza dell'oggetto.")
        FP += 1
        
        logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
        token_ids_generati = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
        
        target_indices = []
        target_token_texts = []
        
        # Cerchiamo tutti i token legati all'allucinazione (la parola YES o il nome dell'oggetto finto)
        for i, t_id in enumerate(token_ids_generati):
            parola_token = processor.decode([t_id]).strip().lower()
            parola_pulita = "".join(c for c in parola_token if c.isalpha())
            
            if parola_pulita == "yes" or (fake_obj.lower() in parola_pulita and len(parola_pulita) > 1):
                target_indices.append(i)
                target_token_texts.append(parola_token)
                
        if not target_indices:
            print("   [?] Nessun token chiave trovato. Uso l'ultima parola del ragionamento come fallback.")
            target_indices.append(len(token_ids_generati) - 1)
            target_token_texts.append(processor.decode([token_ids_generati[-1]]).strip())

        print(f"   -> Estraggo {len(target_indices)} TAM sulle attivazioni dell'allucinazione.")

        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)

        # Creiamo una mappa per ogni parola in cui ha allucinato
        for t_idx, t_text in zip(target_indices, target_token_texts):
            base_name, ext = os.path.splitext(row['path'])
            nome_tam = f"{base_name}_hallucination_step{t_idx}_{fake_obj.replace(' ', '_')}{ext}"
            percorso_salvataggio = os.path.join(tam_dir, nome_tam)
            
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_salvataggio, t_idx, [], False)

            risultati.append({
                'img_id': img_id, 'fake_obj': fake_obj, 'prompt': prompt, 'output': testo_generato.replace('\n', ' '), 
                'target_token': t_text, 'token_index': t_idx, 'obj_iou': 0.0, 'is_hallucination': 1, 'is_correct': 0
            })

            if 'heatmap' in locals(): del heatmap
            gc.collect()

        # Pulizia profonda
        del inputs, image_inputs, video_inputs, outputs, logits, generated_ids
        gc.collect()

    df_temp = pd.DataFrame(risultati)
    df_temp.to_csv(results_file, index=False)

# --- CALCOLO METRICHE GLOBALI (NEGATIVE QUERIES COT) ---
print("\n" + "="*50)
print("🎯 REPORT METRICHE: NEGATIVE QUERIES CON COT (HALLUCINATION TEST)")
print("="*50)

totale_immagini = TN + FP
Specificity = TN / totale_immagini if totale_immagini > 0 else 0
Hallucination_Rate = FP / totale_immagini if totale_immagini > 0 else 0

dati_statistici = [
    {'img_id': '---', 'fake_obj': '--- STATISTICHE ---', 'prompt': '---', 'output': '---', 'target_token': '---', 'token_index': '---', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''},
    {'img_id': 'Totale Immagini Testate', 'fake_obj': str(totale_immagini), 'prompt': '', 'output': '', 'target_token': '', 'token_index': '', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''},
    {'img_id': 'True Negatives (TN) - Resiste', 'fake_obj': str(TN), 'prompt': '', 'output': '', 'target_token': '', 'token_index': '', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''},
    {'img_id': 'False Positives (FP) - Allucinazioni', 'fake_obj': str(FP), 'prompt': '', 'output': '', 'target_token': '', 'token_index': '', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''},
    {'img_id': 'Specificity (Accuratezza)', 'fake_obj': f"{Specificity:.4f}", 'prompt': '', 'output': '', 'target_token': '', 'token_index': '', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''},
    {'img_id': 'Hallucination Rate', 'fake_obj': f"{Hallucination_Rate:.4f}", 'prompt': '', 'output': '', 'target_token': '', 'token_index': '', 'obj_iou': '', 'is_hallucination': '', 'is_correct': ''}
]

risultati_completi = risultati + dati_statistici
df_finale = pd.DataFrame(risultati_completi)
df_finale.to_csv(results_file, index=False)

# --- CREAZIONE DEL FILE ZIP ---
nome_prompt = "Negative_Queries_CoT"
suffisso_img = f"_{len(df)}img" if args.num_images else "_Full"
nome_zip = f"Test_{nome_prompt}{suffisso_img}"

cartella_padre = os.path.dirname(os.path.normpath(output_dir))
percorso_zip_completo = os.path.join(cartella_padre, nome_zip)

print(f"📦 Sto impacchettando i risultati in: {nome_zip}.zip ...")
shutil.make_archive(percorso_zip_completo, 'zip', output_dir)

print("="*50)
print(f"[SUCCESSO] L'archivio {nome_zip}.zip è pronto!")
print(f"📊 Specificity CoT: {Specificity:.2%} | Allucinazioni: {Hallucination_Rate:.2%}")
print("="*50 + "\n")