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
parser = argparse.ArgumentParser(description="VQA Misto (75% Positivi, 25% Negativi) con Energy-based Pointing.")
parser.add_argument('-n', '--num_images', type=int, default=None)
args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_mixed_hard_energy/'
tam_dir = os.path.join(output_dir, 'tam_heatmaps')
os.makedirs(tam_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'results_mixed_hard_energy.csv') 

print("Inizializzo COCO...")
coco = COCO(ann_file)
df = pd.read_csv(csv_path)

cats = coco.loadCats(coco.getCatIds())
tutte_le_categorie = {cat['id']: cat['name'] for cat in cats}

if args.num_images is not None:
    df = df.head(args.num_images)

print("Carico il modello Qwen2-VL...")
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=True)

special_ids = {'img_id': [151652, 151653], 'prompt_id': [151653, [151645, 198, 151644, 77091]], 'answer_id': [[198, 151644, 77091, 198], -1]}

# Parametri Energy-based Pointing
SOGLIA_SALIENZA = 0.30  # Gli hotspot devono avere un'intensità >= 30% del picco massimo
SOGLIA_EIG = 0.20       # Almeno il 20% dell'energia totale deve cadere nell'oggetto
MAX_HOTSPOTS = 2        # Limite massimo di dispersione

risultati = []

# Tracker Metriche
Pos_TP = 0  
Pos_FP_Scattered = 0  # Troppi hotspot
Pos_FP_Mispointed = 0 # Picco massimo fuori dalla GT
Pos_FP_LowEnergy = 0  # EIG < 20%
Pos_FN = 0  
Tot_Pos = 0

Neg_TN = 0  
Neg_FP_Alluc = 0  
Tot_Neg = 0

lista_eig_positive = []

# --- CICLO SULLE IMMAGINI ---
for index, row in df.iterrows():
    img_id = row['img_id']
    img_path = os.path.join(output_dir, row['path'])
    obj_main = row['obj_main']
    ann_id_main = row['ann_id_main']
    
    # 75% Positivo / 25% Negativo
    is_positive = random.random() < 0.75
    
    if is_positive:
        target_obj = obj_main
        query_type = "Positive"
        Tot_Pos += 1
    else:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cat_presenti = set([a['category_id'] for a in anns])
        cat_assenti = [c for c in tutte_le_categorie.keys() if c not in cat_presenti]
        target_obj = tutte_le_categorie[random.choice(cat_assenti)]
        query_type = "Negative"
        Tot_Neg += 1
        
    print(f"\n[{index+1}/{len(df)}] Img: {row['path']} | Query: {query_type} | Target: {target_obj}")
    
    prompt = f"Is there a {target_obj}?Is there a {target_obj}?"
    system_prompt = "You are an expert computer vision model trained to accurately identify objects in images."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}
    ]    
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    testo_pulito = testo_generato.lower()
    
    # GATE 1: Controllo Semantico
    ha_confermato = "yes" in testo_pulito or (target_obj.lower() in testo_pulito and len(target_obj) > 1)
    ha_negato = "no" in testo_pulito and "yes" not in testo_pulito
    
    # Variabili di default per il CSV
    is_correct_id = 0
    eig = 0.0
    num_hotspots = 0
    target_token_text = "NONE"
    target_idx = -1
    failure_reason = "None"
    
    if query_type == "Positive":
        if ha_negato or not ha_confermato:
            print("   [-] FN: Il modello non ha confermato verbalmente l'oggetto (Fallito Gate 1).")
            failure_reason = "FN_Text"
            Pos_FN += 1
        else:
            # Trova token e calcola TAM
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            maschera_coco = coco.annToMask(coco.loadAnns(ann_id_main)[0]) 
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_pos_{target_obj.replace(' ', '_')}.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            heatmap_norm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if len(heatmap.shape) == 3 else heatmap.astype(np.float32)
            
            # --- ENERGY-BASED POINTING LOGIC ---
            
            # GATE 2: Calcolo Hotspots (Scattering Check)
            max_val = np.max(heatmap_norm)
            _, binary_map = cv2.threshold((heatmap_norm * 255).astype(np.uint8), int(SOGLIA_SALIENZA * max_val * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_hotspots = len(contours)
            
            if num_hotspots > MAX_HOTSPOTS:
                print(f"   [!] FP: Mappa troppo dispersa ({num_hotspots} hotspots). Fallito Gate 2.")
                failure_reason = "FP_Scattered"
                Pos_FP_Scattered += 1
            else:
                # GATE 3: Pointing Game
                _, _, _, max_loc = cv2.minMaxLoc(heatmap_norm)
                max_x, max_y = max_loc
                
                if maschera_coco[max_y, max_x] == 0:
                    print(f"   [!] FP: Pointing Game fallito. Il picco massimo ({max_x}, {max_y}) è fuori dall'oggetto. Fallito Gate 3.")
                    failure_reason = "FP_Mispointed"
                    Pos_FP_Mispointed += 1
                else:
                    # GATE 4: EIG (Energy Inside Ground-Truth)
                    energy_inside = np.sum(heatmap_norm[maschera_coco == 1])
                    energy_total = np.sum(heatmap_norm)
                    eig = energy_inside / energy_total if energy_total > 0 else 0
                    
                    lista_eig_positive.append(eig)
                    
                    if eig < SOGLIA_EIG:
                        print(f"   [!] FP: Energia insufficiente (EIG: {eig:.2f} < {SOGLIA_EIG}). Fallito Gate 4.")
                        failure_reason = "FP_LowEnergy"
                        Pos_FP_LowEnergy += 1
                    else:
                        # SUPERATI TUTTI I GATE
                        print(f"   [+] TP! Puntamento corretto ed Energia valida (EIG: {eig:.2f}, Hotspots: {num_hotspots})")
                        is_correct_id = 1
                        failure_reason = "None"
                        Pos_TP += 1
                
    elif query_type == "Negative":
        if ha_negato or not ha_confermato:
            print("   [+] TN: Resiste all'allucinazione.")
            is_correct_id = 1 
            failure_reason = "None"
            Neg_TN += 1
        else:
            print("   [!] FP_Alluc: Allucinazione!")
            failure_reason = "FP_Hallucination"
            Neg_FP_Alluc += 1
            
            # Estraiamo la mappa per registrare gli hotspot dell'allucinazione
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_hallucination_{target_obj.replace(' ', '_')}.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            
            heatmap_norm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if len(heatmap.shape) == 3 else heatmap.astype(np.float32)
            max_val = np.max(heatmap_norm)
            _, binary_map = cv2.threshold((heatmap_norm * 255).astype(np.uint8), int(SOGLIA_SALIENZA * max_val * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_hotspots = len(contours)

    # Salvataggio riga CSV omologata al nuovo sistema
    risultati.append({
        'img_id': img_id, 'query_type': query_type, 'prompt': prompt, 
        'output': testo_generato.replace('\n', ' '), 'target_token': target_token_text, 
        'token_index': target_idx, 'num_hotspots': num_hotspots, 'eig': round(eig, 4), 
        'is_correct_id': is_correct_id, 'failure_reason': failure_reason
    })

    # PULIZIA ESTREMA CPU
    del inputs, image_inputs, video_inputs, outputs, generated_ids
    if 'logits' in locals(): del logits
    if 'heatmap' in locals(): del heatmap
    if 'heatmap_norm' in locals(): del heatmap_norm
    if 'binary_map' in locals(): del binary_map
    if 'maschera_coco' in locals(): del maschera_coco
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(risultati).to_csv(results_file, index=False)


# --- CALCOLO METRICHE GLOBALI (ENERGY-BASED) ---
print("\n" + "="*60)
print("🎯 REPORT METRICHE: ENERGY-BASED POINTING (75 Pos / 25 Neg)")
print("="*60)

Sensitivity = Pos_TP / Tot_Pos if Tot_Pos > 0 else 0
Specificity = Neg_TN / Tot_Neg if Tot_Neg > 0 else 0
Hallucination_Rate = Neg_FP_Alluc / Tot_Neg if Tot_Neg > 0 else 0
m_EIG = np.mean(lista_eig_positive) if lista_eig_positive else 0

print(f"✅ True Positives (TP): {Pos_TP}")
print(f"❌ Falsi Positivi Totali su Oggetti Presenti: {Pos_FP_Scattered + Pos_FP_Mispointed + Pos_FP_LowEnergy}")
print(f"   ├─ Mappe Disperse (> {MAX_HOTSPOTS} hotspot): {Pos_FP_Scattered}")
print(f"   ├─ Pointing Sbagliato (Picco fuori GT): {Pos_FP_Mispointed}")
print(f"   └─ Bassa Energia (EIG < {SOGLIA_EIG}): {Pos_FP_LowEnergy}")
print(f"❌ Falsi Negativi (FN testuali): {Pos_FN}")
print("-" * 30)
print(f"✅ True Negatives (Resistenza Allucinazione): {Neg_TN}")
print(f"❌ Falsi Positivi (Allucinazioni Pure): {Neg_FP_Alluc}")
print("-" * 30)
print(f"📊 Sensitivity (Recall): {Sensitivity:.2%}")
print(f"📊 Specificity: {Specificity:.2%}")
print(f"📊 Energia Media (EIG) dei TP: {m_EIG:.4f}")

# Creazione ZIP finale...
nome_prompt = "Mixed_HardPrompt_Energy"
suffisso_img = f"_{len(df)}img" if args.num_images else "_Full"
nome_zip = f"Test_{nome_prompt}{suffisso_img}"
shutil.make_archive(os.path.join(os.path.dirname(os.path.normpath(output_dir)), nome_zip), 'zip', output_dir)
print(f"\n[SUCCESSO] L'archivio {nome_zip}.zip e il CSV sono pronti!")