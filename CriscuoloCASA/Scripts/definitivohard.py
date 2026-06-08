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

# --- 1. IMPOSTAZIONE SEED GLOBALE (RIPRODUCIBILITÀ) ---
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# -----------------------------------------------------

# --- CONFIGURAZIONE ARGOMENTI DA RIGA DI COMANDO (CLI) ---
parser = argparse.ArgumentParser(description="VQA Misto con Object-Priority Token, Reverse Jet, K-Means Clustering e TBR.")
parser.add_argument('-n', '--num_images', type=int, default=None)
args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_mixed_hard_distractor/'
vis_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'results_mixed_hard_definitivo.csv') 

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
SOGLIA_TBR = 1.0 #Deve vedere più energia dentro l'oggetto che fuori per essere considerato un TP valido 
MIN_HOTSPOT_AREA = 200  
MAX_EXTERNAL_DISTRACTORS = 0 
K_CLUSTERS = 3  # Divisione in Sfondo, Alone (Penombra), e Core (Picco)

risultati = []

# Tracker Metriche
Pos_TP = 0; Pos_TP_Fallback = 0; Pos_FP_Scattered = 0; Pos_FP_Mispointed = 0; Pos_FP_LowEnergy = 0; Pos_FN = 0; Tot_Pos = 0
Neg_TN = 0; Neg_FP_Alluc = 0; Tot_Neg = 0
lista_tbr_positive = []

# --- CICLO SULLE IMMAGINI ---
for index, row in df.iterrows():
    img_id = row['img_id']
    img_path = os.path.join(output_dir, row['path'])
    obj_main = row['obj_main']
    ann_id_main = row['ann_id_main']
    img_base_name = os.path.splitext(os.path.basename(row['path']))[0]
    
    img_output_dir = os.path.join(vis_dir, f"img_{img_base_name}")
    os.makedirs(img_output_dir, exist_ok=True)
    
    orig_img = cv2.imread(img_path)
    if orig_img is not None:
        cv2.imwrite(os.path.join(img_output_dir, "0_original.jpg"), orig_img)
        orig_h, orig_w = orig_img.shape[:2]
    else:
        orig_h, orig_w = 480, 640 
    
    is_positive = random.random() < 0.75
    if is_positive:
        target_obj = obj_main; query_type = "Positive"; Tot_Pos += 1
    else:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cat_presenti = set([a['category_id'] for a in anns])
        cat_assenti = [c for c in tutte_le_categorie.keys() if c not in cat_presenti]
        target_obj = tutte_le_categorie[random.choice(cat_assenti)]; query_type = "Negative"; Tot_Neg += 1
        
    print(f"\n[{index+1}/{len(df)}] Img: {row['path']} | Query: {query_type} | Target: {target_obj}")
    
    prompt = f"Is there a {target_obj}?  If you are confident that the object is present, answer with 'Yes.There is a {target_obj} in the image.' If you are confident that the object is not present, answer with 'No. There is no {target_obj} in the image.'."
    system_prompt = "You are an expert computer vision model trained to accurately identify objects in images. You will be asked to determine the presence of a specific object in the image. Be precise and avoid hallucinations. Focus on the visual evidence in the image to make your decision. Always respect the given format for the answers."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}
    ]    
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # --- 2. ALLINEAMENTO PARAMETRI: do_sample=False, rimossa temperature ---
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    testo_pulito = testo_generato.lower()
    
    print(f"   [💬] Risposta Modello: '{testo_generato}'")
    
    ha_confermato = "yes" in testo_pulito or (target_obj.lower() in testo_pulito and len(target_obj) > 1)
    ha_negato = "no" in testo_pulito and "yes" not in testo_pulito
    
    is_correct_id = 0; tbr = 0.0; num_hotspots = 0; num_distractors = 0; target_idx = -1; soglia_kmeans_norm = 0.0; failure_reason = "None"
    target_token_text = "NONE"
    
    if query_type == "Positive":
        if ha_negato or not ha_confermato:
            print("   [-] FN: Il modello ha rifiutato l'oggetto presente.")
            failure_reason = "FN_Text"; Pos_FN += 1
        else:
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            token_obj_idx = -1
            token_yes_idx = -1
            
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if target_obj.lower() in parola and token_obj_idx == -1:
                    token_obj_idx = i
                if "yes" in parola and token_yes_idx == -1:
                    token_yes_idx = i
            
            if token_obj_idx != -1: target_idx = token_obj_idx
            elif token_yes_idx != -1: target_idx = token_yes_idx
            else: target_idx = 0
                
            target_token_text = processor.decode([token_ids[target_idx]]).strip()
            print(f"   [🔍] Token Selezionato per TAM: '{target_token_text}' (Indice: {target_idx})")
            
            maschera_coco = coco.annToMask(coco.loadAnns(ann_id_main)[0]) 
            cv2.imwrite(os.path.join(img_output_dir, "4_ground_truth.png"), maschera_coco * 255)
            
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(img_output_dir, "1_heatmap_tam.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            
            if len(heatmap.shape) == 3:
                hsv_map = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
                hue = hsv_map[:, :, 0].astype(np.float32)
                hue[hue > 120] = 0 
                heatmap_norm = np.clip((120.0 - hue) / 120.0, 0.0, 1.0)
            else:
                heatmap_raw = heatmap.astype(np.float32)
                h_min, h_max = np.min(heatmap_raw), np.max(heatmap_raw)
                heatmap_norm = (heatmap_raw - h_min) / (h_max - h_min + 1e-8)

            heatmap_norm = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # =================================================================
            # K-MEANS CLUSTERING 
            # =================================================================
            heatmap_8bit = (heatmap_norm * 255).astype(np.uint8) 
            pixel_values = heatmap_8bit.reshape((-1, 1)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            _, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            core_cluster_idx = np.argmax(centers)
            
            binary_map_raw = np.zeros_like(heatmap_8bit)
            binary_map_raw[labels.reshape(heatmap_8bit.shape) == core_cluster_idx] = 255
            
            cv2.imwrite(os.path.join(img_output_dir, "2_kmeans_raw.png"), binary_map_raw)
            
            min_val_in_core = np.min(pixel_values[labels == core_cluster_idx])
            soglia_kmeans_norm = min_val_in_core / 255.0
            
            # --- MORFOLOGIA E FILTRO AREA ---
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary_map_morph = cv2.morphologyEx(binary_map_raw, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(binary_map_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_HOTSPOT_AREA]
            
            # --- GATE 2: SEPARAZIONE TARGET VS DISTRACTOR ---
            target_hotspots = []
            distractor_hotspots = []
            
            for c in valid_contours:
                c_mask = np.zeros_like(binary_map_morph)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                
                if np.any((c_mask == 255) & (maschera_coco == 1)):
                    target_hotspots.append(c)
                else:
                    distractor_hotspots.append(c)
                    
            num_hotspots = len(target_hotspots)
            num_distractors = len(distractor_hotspots)
            
            clean_binary_map = np.zeros_like(binary_map_morph)
            cv2.drawContours(clean_binary_map, target_hotspots, -1, 255, -1)
            cv2.imwrite(os.path.join(img_output_dir, "3_hotspots_clean.png"), clean_binary_map) 
            
            _, _, _, max_loc = cv2.minMaxLoc(heatmap_norm)
            max_x, max_y = max_loc; cX, cY = max_x, max_y
            used_fallback = False
            
            if num_distractors > MAX_EXTERNAL_DISTRACTORS:
                print(f"   [!] FP: Modello distratto. ({num_distractors} hotspot fuori bersaglio).")
                failure_reason = "FP_Scattered"; Pos_FP_Scattered += 1
            else:
                if num_hotspots == 0:
                    print(f"   [⚠️] Nessun hotspot tocca il target. Fallback su Max Assoluto.")
                    used_fallback = True
                else:
                    best_contour = target_hotspots[0] 
                    max_energy = -1
                    for c in target_hotspots:
                        temp_mask = np.zeros_like(binary_map_morph)
                        cv2.drawContours(temp_mask, [c], -1, 255, -1)
                        energy = np.sum(heatmap_norm[temp_mask == 255])
                        if energy > max_energy:
                            max_energy = energy; best_contour = c
                    
                    main_hotspot_mask = np.zeros_like(binary_map_morph)
                    cv2.drawContours(main_hotspot_mask, [best_contour], -1, 255, -1)
                    heatmap_isolated = np.zeros_like(heatmap_norm)
                    heatmap_isolated[main_hotspot_mask == 255] = heatmap_norm[main_hotspot_mask == 255]
                    
                    M = cv2.moments(heatmap_isolated)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        used_fallback = True
                
                # --- VISUALIZZAZIONE DEBUG ---
                debug_vis = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
                debug_vis = cv2.addWeighted(debug_vis, 0.7, np.zeros_like(debug_vis), 0, 0)
                
                gt_contours, _ = cv2.findContours((maschera_coco*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_vis, gt_contours, -1, (0, 255, 0), 2)
                
                cv2.drawContours(debug_vis, target_hotspots, -1, (0, 0, 255), 2)
                if num_distractors > 0:
                    cv2.drawContours(debug_vis, distractor_hotspots, -1, (255, 0, 255), 2)
                    
                cross_color = (0, 255, 255) if used_fallback else (255, 0, 0)
                cv2.drawMarker(debug_vis, (cX, cY), cross_color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.imwrite(os.path.join(img_output_dir, "5_pointing_debug.jpg"), debug_vis)
                
                # GATE 3: Pointing Game
                pointing_success = False
                if maschera_coco[cY, cX] == 1:
                    pointing_success = True
                else:
                    if num_hotspots > 0:
                        if maschera_coco[max_y, max_x] == 1:
                            pointing_success = True; cX, cY = max_x, max_y; used_fallback = True
                
                if not pointing_success:
                    print(f"   [!] FP: Puntamento geometrico esterno all'oggetto.")
                    failure_reason = "FP_Mispointed"; Pos_FP_Mispointed += 1
                else:
                    # GATE 4: TBR calcolato sull'hotspot
                    mask_out = (maschera_coco == 0)
                    mu_out = np.mean(heatmap_norm[mask_out]) if np.any(mask_out) else 0.0
                    mask_in_hotspot = (maschera_coco == 1) & (clean_binary_map == 255)
                    
                    if num_hotspots == 0 or not np.any(mask_in_hotspot):
                        mask_in = (maschera_coco == 1)
                        mu_in = np.mean(heatmap_norm[mask_in]) if np.any(mask_in) else 0.0
                    else:
                        mu_in = np.mean(heatmap_norm[mask_in_hotspot])

                    tbr = mu_in / (mu_out + 1e-6)
                    lista_tbr_positive.append(tbr)
                    
                    if tbr <= SOGLIA_TBR:
                        print(f"   [!] FP: Contrasto insufficiente (TBR: {tbr:.2f} < {SOGLIA_TBR}).")
                        failure_reason = "FP_LowEnergy"; Pos_FP_LowEnergy += 1
                    else:
                        is_correct_id = 1; failure_reason = "None"
                        if used_fallback: Pos_TP_Fallback += 1
                        else: Pos_TP += 1
                        print(f"   [+] TP Convalidato! (TBR: {tbr:.2f}, Target Hotspots: {num_hotspots}, Fallback: {used_fallback})")
                        
    elif query_type == "Negative":
        if ha_negato or not ha_confermato:
            print("   [+] TN: Il modello ha respinto l'oggetto falso.")
            is_correct_id = 1; failure_reason = "None"; Neg_TN += 1
        else:
            print("   [!] FP_Alluc: Il modello ha confermato l'oggetto assente!")
            failure_reason = "FP_Hallucination"; Neg_FP_Alluc += 1
            
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            token_obj_idx = -1
            token_yes_idx = -1
            
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if target_obj.lower() in parola and token_obj_idx == -1:
                    token_obj_idx = i
                if "yes" in parola and token_yes_idx == -1:
                    token_yes_idx = i
            
            if token_obj_idx != -1: target_idx = token_obj_idx
            elif token_yes_idx != -1: target_idx = token_yes_idx
            else: target_idx = 0
                
            target_token_text = processor.decode([token_ids[target_idx]]).strip()
            
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(img_output_dir, "1_heatmap_tam.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            
            if len(heatmap.shape) == 3:
                hsv_map = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
                hue = hsv_map[:, :, 0].astype(np.float32)
                hue[hue > 120] = 0 
                heatmap_norm = np.clip((120.0 - hue) / 120.0, 0.0, 1.0)
            else:
                heatmap_raw = heatmap.astype(np.float32)
                h_min, h_max = np.min(heatmap_raw), np.max(heatmap_raw)
                heatmap_norm = (heatmap_raw - h_min) / (h_max - h_min + 1e-8)
            heatmap_norm = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            heatmap_8bit = (heatmap_norm * 255).astype(np.uint8)
            
            pixel_values = heatmap_8bit.reshape((-1, 1)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            core_cluster_idx = np.argmax(centers)
            
            binary_map_raw = np.zeros_like(heatmap_8bit)
            binary_map_raw[labels.reshape(heatmap_8bit.shape) == core_cluster_idx] = 255
            cv2.imwrite(os.path.join(img_output_dir, "2_kmeans_raw.png"), binary_map_raw)
            
            min_val_in_core = np.min(pixel_values[labels == core_cluster_idx])
            soglia_kmeans_norm = min_val_in_core / 255.0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary_map_morph = cv2.morphologyEx(binary_map_raw, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(binary_map_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_HOTSPOT_AREA]
            num_hotspots = 0 
            num_distractors = len(valid_contours)  
            
            clean_binary_map = np.zeros_like(binary_map_morph)
            cv2.drawContours(clean_binary_map, valid_contours, -1, 255, -1)
            cv2.imwrite(os.path.join(img_output_dir, "3_hotspots_clean.png"), clean_binary_map)

    risultati.append({
        'img_id': img_id, 'query_type': query_type, 'prompt': prompt, 
        'output': testo_generato.replace('\n', ' '), 'target_token': target_token_text, 
        'token_index': target_idx, 'num_target_hotspots': num_hotspots, 'num_distractors': num_distractors,
        'kmeans_threshold': round(soglia_kmeans_norm, 4), 'tbr': round(tbr, 4), 
        'is_correct_id': is_correct_id, 'failure_reason': failure_reason
    })

    del inputs, image_inputs, video_inputs, outputs, generated_ids
    for var in ['logits', 'heatmap', 'heatmap_raw', 'heatmap_norm', 'heatmap_8bit', 'binary_map_raw', 'binary_map_morph', 'clean_binary_map', 'maschera_coco', 'orig_img', 'hsv_map', 'hue', 'heatmap_isolated', 'main_hotspot_mask', 'debug_vis', 'gt_contours', 'c_mask', 'target_hotspots', 'distractor_hotspots', 'pixel_values', 'labels', 'centers']:
        if var in locals(): del locals()[var]
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(risultati).to_csv(results_file, index=False)

# --- REPORT FINALE ---
print("\n" + "="*60)
print("🎯 REPORT ESPERIMENTO: K-MEANS CLUSTERING PIPELINE")
print("="*60)
Tot_TP_Effettivi = Pos_TP + Pos_TP_Fallback
Sensitivity = Tot_TP_Effettivi / Tot_Pos if Tot_Pos > 0 else 0
Specificity = Neg_TN / Tot_Neg if Tot_Neg > 0 else 0
m_TBR = np.mean(lista_tbr_positive) if lista_tbr_positive else 0

print(f"✅ True Positives (TP) Totali: {Tot_TP_Effettivi}")
print(f"   ├─ Tramite Centroide Diretto: {Pos_TP}")
print(f"   └─ Salvati da Fallback: {Pos_TP_Fallback}")
print(f"❌ Falsi Positivi (FP) su Oggetti Presenti: {Pos_FP_Scattered + Pos_FP_Mispointed + Pos_FP_LowEnergy}")
print(f"   ├─ Modello Distratto (Hotspot esterni alla GT): {Pos_FP_Scattered}")
print(f"   ├─ Puntamento Fallito (Centroide esterno): {Pos_FP_Mispointed}")
print(f"   └─ Contrasto TBR Insufficiente: {Pos_FP_LowEnergy}")
print(f"❌ Falsi Negativi (FN testuali): {Pos_FN}")
print("-" * 40)
print(f"✅ True Negatives (TN - Rifiuto Allucinazione): {Neg_TN}")
print(f"❌ Falsi Positivi (FP_alluc - Allucinazione pura): {Neg_FP_Alluc}")
print("-" * 40)
print(f"📊 Sensitivity: {Sensitivity:.2%}")
print(f"📊 Specificity: {Specificity:.2%}")
print(f"📊 TBR Medio TP: {m_TBR:.4f}")

nome_prompt = "Mixed_Hard_KMeans_Pipeline_FullPrompt"
shutil.make_archive(os.path.join(os.path.dirname(os.path.normpath(output_dir)), f"Test_{nome_prompt}"), 'zip', output_dir)
print(f"\n[SUCCESSO] Archiviato! Controlla la cartella 'visualizations' nell'output.")