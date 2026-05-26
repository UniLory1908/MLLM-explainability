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
from metrics import calcola_metriche_tam, calcola_pixel_accuracy

# --- CONFIGURAZIONE ARGOMENTI DA RIGA DI COMANDO (CLI) ---
parser = argparse.ArgumentParser(description="VQA Misto (75-25) con TBR (Target-to-Background Ratio).")
parser.add_argument('-n', '--num_images', type=int, default=None)
args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_mixed_hard_tbr/'
tam_dir = os.path.join(output_dir, 'tam_heatmaps')
os.makedirs(tam_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'results_mixed_hard_tbr.csv') 

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

# --- PARAMETRI METODOLOGICI ---
SOGLIA_SALIENZA = 0.30  
SOGLIA_TBR = 2.0        # L'intensità media dell'hotspot interno deve essere >= 2x rispetto al rumore di fondo
MAX_HOTSPOTS = 2        

risultati = []

# Tracker Metriche
Pos_TP = 0  
Pos_TP_Fallback = 0  
Pos_FP_Scattered = 0  
Pos_FP_Mispointed = 0 
Pos_FP_LowEnergy = 0  
Pos_FN = 0  
Tot_Pos = 0

Neg_TN = 0  
Neg_FP_Alluc = 0  
Tot_Neg = 0

lista_tbr_positive = []

# --- CICLO SULLE IMMAGINI ---
for index, row in df.iterrows():
    img_id = row['img_id']
    img_path = os.path.join(output_dir, row['path'])
    obj_main = row['obj_main']
    ann_id_main = row['ann_id_main']
    
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
    
    ha_confermato = "yes" in testo_pulito or (target_obj.lower() in testo_pulito and len(target_obj) > 1)
    ha_negato = "no" in testo_pulito and "yes" not in testo_pulito
    
    is_correct_id = 0
    tbr = 0.0
    num_hotspots = 0
    target_token_text = "NONE"
    target_idx = -1
    failure_reason = "None"
    
    if query_type == "Positive":
        if ha_negato or not ha_confermato:
            print("   [-] FN: Il modello ha rifiutato l'oggetto presente (Fallito Gate 1).")
            failure_reason = "FN_Text"
            Pos_FN += 1
        else:
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            maschera_coco = coco.annToMask(coco.loadAnns(ann_id_main)[0]) 
            orig_h, orig_w = maschera_coco.shape  
            
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_pos_tbr.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            heatmap_norm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if len(heatmap.shape) == 3 else heatmap.astype(np.float32)
            
            heatmap_norm = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # GATE 2: Scattering Check
            max_val = np.max(heatmap_norm)
            _, binary_map = cv2.threshold((heatmap_norm * 255).astype(np.uint8), int(SOGLIA_SALIENZA * max_val * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_hotspots = len(contours)
            
            _, _, _, max_loc = cv2.minMaxLoc(heatmap_norm)
            max_x, max_y = max_loc
            
            used_fallback = False
            
            if num_hotspots > MAX_HOTSPOTS:
                print(f"   [!] FP: Mappa troppo diffusa ({num_hotspots} hotspots). Fallito Gate 2.")
                failure_reason = "FP_Scattered"
                Pos_FP_Scattered += 1
            else:
                if num_hotspots == 0:
                    print(f"   [⚠️] Nessun hotspot. Attivo Fallback sul Massimo Assoluto...")
                    cX, cY = max_x, max_y
                    used_fallback = True
                else:
                    best_contour = contours[0] 
                    max_energy = -1
                    for c in contours:
                        temp_mask = np.zeros_like(binary_map)
                        cv2.drawContours(temp_mask, [c], -1, 255, -1)
                        energy = np.sum(heatmap_norm[temp_mask == 255])
                        if energy > max_energy:
                            max_energy = energy
                            best_contour = c
                    
                    main_hotspot_mask = np.zeros_like(binary_map)
                    cv2.drawContours(main_hotspot_mask, [best_contour], -1, 255, -1)
                    heatmap_isolated = np.zeros_like(heatmap_norm)
                    heatmap_isolated[main_hotspot_mask == 255] = heatmap_norm[main_hotspot_mask == 255]
                    
                    M = cv2.moments(heatmap_isolated)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = max_x, max_y
                        used_fallback = True
                
                # GATE 3: Pointing Game
                pointing_success = False
                
                if maschera_coco[cY, cX] == 1:
                    pointing_success = True
                    if num_hotspots == 0:
                        print(f"   [->] Puntamento tramite Massimo Assoluto ({cX}, {cY}) DENTRO la GT.")
                    else:
                        print(f"   [->] Centroide Localizzato ({cX}, {cY}) DENTRO la GT.")
                else:
                    if num_hotspots > 0:
                        print(f"   [⚠️] Centroide ({cX}, {cY}) FUORI dalla GT. Provo Fallback secondario sul Massimo Assoluto...")
                        if maschera_coco[max_y, max_x] == 1:
                            pointing_success = True
                            cX, cY = max_x, max_y
                            used_fallback = True
                            print(f"   [->] Fallback Riuscito! Il picco massimo ({max_x}, {max_y}) è DENTRO la GT.")
                
                if not pointing_success:
                    print(f"   [!] FP: Sia il punto selezionato che il massimo sono esterni all'oggetto. Fallito Gate 3.")
                    failure_reason = "FP_Mispointed"
                    Pos_FP_Mispointed += 1
                else:
                    # GATE 4: Confidenza Visiva (TBR - Target-to-Background Ratio)
                    
                    # 1. Media Esterna (Sfondo totale)
                    mask_out = (maschera_coco == 0)
                    mu_out = np.mean(heatmap_norm[mask_out]) if np.any(mask_out) else 0.0
                    
                    # 2. Media Interna (Hotspot dentro la Maschera)
                    mask_in_hotspot = (maschera_coco == 1) & (binary_map == 255)
                    
                    if num_hotspots == 0 or not np.any(mask_in_hotspot):
                        # Fallback all'intera area della GT se non c'è hotspot o se l'hotspot è totalmente fuori
                        mask_in = (maschera_coco == 1)
                        mu_in = np.mean(heatmap_norm[mask_in]) if np.any(mask_in) else 0.0
                        print("      (Calcolo mu_in sull'intera area della Ground Truth)")
                    else:
                        # Logica Utente: media calcolata solo sulla porzione di hotspot interna alla GT
                        mu_in = np.mean(heatmap_norm[mask_in_hotspot])
                        print("      (Calcolo mu_in focalizzato solo sull'hotspot interno)")

                    # Calcolo TBR (protezione divisione per zero)
                    tbr = mu_in / (mu_out + 1e-6)
                    lista_tbr_positive.append(tbr)
                    
                    if tbr < SOGLIA_TBR:
                        print(f"   [!] FP: Contrasto insufficiente (TBR: {tbr:.2f} < {SOGLIA_TBR}). Fallito Gate 4.")
                        failure_reason = "FP_LowEnergy"
                        Pos_FP_LowEnergy += 1
                    else:
                        is_correct_id = 1
                        failure_reason = "None"
                        if used_fallback: Pos_TP_Fallback += 1
                        else: Pos_TP += 1
                        print(f"   [+] TP Convalidato! (TBR: {tbr:.2f}, Hotspots: {num_hotspots}, Fallback: {used_fallback})")
                        
    elif query_type == "Negative":
        if ha_negato or not ha_confermato:
            print("   [+] TN: Il modello ha respinto l'oggetto falso.")
            is_correct_id = 1
            failure_reason = "None"
            Neg_TN += 1
        else:
            print("   [!] FP_Alluc: Il modello ha confermato l'oggetto assente!")
            failure_reason = "FP_Hallucination"
            Neg_FP_Alluc += 1
            
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                orig_h, orig_w = orig_img.shape[:2]
            else:
                orig_h, orig_w = 480, 640 
                
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_hallucination.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            heatmap_norm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if len(heatmap.shape) == 3 else heatmap.astype(np.float32)
            
            heatmap_norm = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            max_val = np.max(heatmap_norm)
            _, binary_map = cv2.threshold((heatmap_norm * 255).astype(np.uint8), int(SOGLIA_SALIENZA * max_val * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_hotspots = len(contours)

    # Nota: la colonna si chiama 'tbr' al posto di 'eig' per chiarezza nel CSV
    risultati.append({
        'img_id': img_id, 'query_type': query_type, 'prompt': prompt, 
        'output': testo_generato.replace('\n', ' '), 'target_token': target_token_text, 
        'token_index': target_idx, 'num_hotspots': num_hotspots, 'tbr': round(tbr, 4), 
        'is_correct_id': is_correct_id, 'failure_reason': failure_reason
    })

    # --- RESET PROFONDO CPU ---
    del inputs, image_inputs, video_inputs, outputs, generated_ids
    if 'logits' in locals(): del logits
    if 'heatmap' in locals(): del heatmap
    if 'heatmap_norm' in locals(): del heatmap_norm
    if 'binary_map' in locals(): del binary_map
    if 'maschera_coco' in locals(): del maschera_coco
    if 'orig_img' in locals(): del orig_img
    if 'heatmap_isolated' in locals(): del heatmap_isolated
    if 'main_hotspot_mask' in locals(): del main_hotspot_mask
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(risultati).to_csv(results_file, index=False)

# --- REPORT FINALE SUL TERMINALE ---
print("\n" + "="*60)
print("🎯 REPORT ESPERIMENTO: POINTING GAME + TBR")
print("="*60)
Tot_TP_Effettivi = Pos_TP + Pos_TP_Fallback
Sensitivity = Tot_TP_Effettivi / Tot_Pos if Tot_Pos > 0 else 0
Specificity = Neg_TN / Tot_Neg if Tot_Neg > 0 else 0
m_TBR = np.mean(lista_tbr_positive) if lista_tbr_positive else 0

print(f"✅ True Positives (TP) Totali: {Tot_TP_Effettivi}")
print(f"   ├─ Tramite Centroide Diretto: {Pos_TP}")
print(f"   └─ Salvati da Fallback: {Pos_TP_Fallback}")
print(f"❌ Falsi Positivi (FP) su Oggetti Presenti: {Pos_FP_Scattered + Pos_FP_Mispointed + Pos_FP_LowEnergy}")
print(f"   ├─ Mappe Diffuse (> {MAX_HOTSPOTS} hotspot): {Pos_FP_Scattered}")
print(f"   ├─ Puntamento Fallito (Esterno alla GT): {Pos_FP_Mispointed}")
print(f"   └─ Contrasto TBR Insufficiente (< {SOGLIA_TBR}): {Pos_FP_LowEnergy}")
print(f"❌ Falsi Negativi (FN testuali): {Pos_FN}")
print("-" * 40)
print(f"✅ True Negatives (TN - Rifiuto Allucinazione): {Neg_TN}")
print(f"❌ Falsi Positivi (FP_alluc - Allucinazione pura): {Neg_FP_Alluc}")
print("-" * 40)
print(f"📊 Sensitivity (Recall): {Sensitivity:.2%}")
print(f"📊 Specificity: {Specificity:.2%}")
print(f"📊 TBR Medio (Contrasto Energetico) dei TP: {m_TBR:.4f}")

nome_prompt = "Mixed_Hard_TBR"
shutil.make_archive(os.path.join(os.path.dirname(os.path.normpath(output_dir)), f"Test_{nome_prompt}"), 'zip', output_dir)
print(f"\n[SUCCESSO] Esecuzione completata! Controlla il file results_mixed_hard_tbr.csv")