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
parser = argparse.ArgumentParser(description="VQA CoT Spaziale con Object-Priority Token, K-Means Clustering e TBR.")
parser.add_argument('-n', '--num_images', type=int, default=None)
args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_cot_spatial_kmeans/'
vis_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'results_cot_definitivo.csv') 

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
SOGLIA_TBR = 1.0        
MIN_HOTSPOT_AREA = 200  
MAX_EXTERNAL_DISTRACTORS = 0 
K_CLUSTERS = 3  

risultati = []

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
    
    # ---------------------------------------------------------
    # PROMPT CoT (Chain of Thought Spaziale)
    # ---------------------------------------------------------
    system_prompt = (
        "You are an expert computer vision model. Your task is to identify objects by first reasoning about the scene. "
        "Always follow the requested steps and use complete sentences."
    )
    prompt = (
        f"Step 1: Look at the image and describe the main objects and their spatial positions.\n"
        f"Step 2: Check if you can see a '{target_obj}' among them.\n"
        f"Step 3: Answer to the question 'Is there a {target_obj}?' and provide a brief justification based on the visual evidence you analyzed."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}
    ]    
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.5, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    testo_pulito = testo_generato.lower()
    
    print(f"   [💬] Risposta Modello (CoT):\n'{testo_generato}'")
    
    ha_confermato = "conclusion: yes" in testo_pulito or (target_obj.lower() in testo_pulito and "no" not in testo_pulito[-15:])
    ha_negato = "conclusion: no" in testo_pulito
    
    # ---------------------------------------------------------
    # RICERCA TOKEN: Identificazione multipla per il CoT
    # ---------------------------------------------------------
    token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
    target_indices = []
    
    for i, t_id in enumerate(token_ids):
        parola = processor.decode([t_id]).strip().lower()
        if target_obj.lower() in parola and len(target_obj) > 1:
            if i not in target_indices: target_indices.append(i)
        if parola in ["yes", "no"] and i > (len(token_ids) - 25): 
            if i not in target_indices: target_indices.append(i)
            
    if not target_indices: target_indices = [len(token_ids) - 1]
    
    # Variabili per l'estrazione delle metriche allineate
    is_correct_id = 0
    image_tp_found = False
    image_failure_reason = "None"
    image_fallback_used = False
    
    best_metrics = {
        'target_token': "NONE",
        'token_index': -1,
        'num_target_hotspots': 0,
        'num_distractors': 0,
        'kmeans_threshold': 0.0,
        'tbr': 0.0
    }
    
    # =========================================================
    # ANALISI POSITIVA (Oggetto Presente)
    # =========================================================
    if query_type == "Positive":
        if ha_negato or not ha_confermato:
            print("   [-] FN: Il modello ha rifiutato l'oggetto presente.")
            image_failure_reason = "FN_Text"
            Pos_FN += 1
        else:
            maschera_coco = coco.annToMask(coco.loadAnns(ann_id_main)[0]) 
            cv2.imwrite(os.path.join(img_output_dir, "4_ground_truth.png"), maschera_coco * 255)
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            
            best_tbr_tracker = -1.0
            
            # --- LOOP SU TUTTI I TOKEN SOSPETTI ---
            for t_idx in target_indices:
                target_token_text = processor.decode([token_ids[t_idx]]).strip()
                print(f"   [🔍] Valuto TAM su Token: '{target_token_text}' (Indice: {t_idx})")
                
                percorso_tam = os.path.join(img_output_dir, f"1_heatmap_tam_tok{t_idx}.jpg")
                heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, t_idx, [], False)
                
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
                cv2.imwrite(os.path.join(img_output_dir, f"2_kmeans_raw_tok{t_idx}.png"), binary_map_raw)
                
                min_val_in_core = np.min(pixel_values[labels == core_cluster_idx])
                soglia_kmeans_norm = float(min_val_in_core) / 255.0
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                binary_map_morph = cv2.morphologyEx(binary_map_raw, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(binary_map_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_HOTSPOT_AREA]
                
                target_hotspots = []
                distractor_hotspots = []
                for c in valid_contours:
                    c_mask = np.zeros_like(binary_map_morph)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    if np.any((c_mask == 255) & (maschera_coco == 1)): target_hotspots.append(c)
                    else: distractor_hotspots.append(c)
                        
                num_hotspots = len(target_hotspots)
                num_distractors = len(distractor_hotspots)
                
                clean_binary_map = np.zeros_like(binary_map_morph)
                cv2.drawContours(clean_binary_map, target_hotspots, -1, 255, -1)
                cv2.imwrite(os.path.join(img_output_dir, f"3_hotspots_clean_tok{t_idx}.png"), clean_binary_map) 
                
                _, _, _, max_loc = cv2.minMaxLoc(heatmap_norm)
                cX, cY = max_loc
                used_fallback = False
                current_failure = "None"
                
                if num_distractors > MAX_EXTERNAL_DISTRACTORS:
                    current_failure = "FP_Scattered"
                else:
                    if num_hotspots == 0:
                        used_fallback = True
                    else:
                        best_contour = target_hotspots[0] 
                        max_energy = -1
                        for c in target_hotspots:
                            temp_mask = np.zeros_like(binary_map_morph)
                            cv2.drawContours(temp_mask, [c], -1, 255, -1)
                            energy = np.sum(heatmap_norm[temp_mask == 255])
                            if energy > max_energy:
                                max_energy = energy
                                best_contour = c
                        
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

                debug_vis = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
                debug_vis = cv2.addWeighted(debug_vis, 0.7, np.zeros_like(debug_vis), 0, 0)
                gt_contours, _ = cv2.findContours((maschera_coco*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_vis, gt_contours, -1, (0, 255, 0), 2)
                cv2.drawContours(debug_vis, target_hotspots, -1, (0, 0, 255), 2)
                if num_distractors > 0: cv2.drawContours(debug_vis, distractor_hotspots, -1, (255, 0, 255), 2)
                    
                cross_color = (0, 255, 255) if used_fallback else (255, 0, 0)
                cv2.drawMarker(debug_vis, (cX, cY), cross_color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.imwrite(os.path.join(img_output_dir, f"5_pointing_debug_tok{t_idx}.jpg"), debug_vis)

                pointing_success = False
                if maschera_coco[cY, cX] == 1: pointing_success = True
                else:
                    if num_hotspots > 0 and maschera_coco[max_loc[1], max_loc[0]] == 1:
                        pointing_success = True
                        cX, cY = max_loc
                        used_fallback = True
                        
                if not pointing_success and current_failure == "None":
                    current_failure = "FP_Mispointed"

                mask_out = (maschera_coco == 0)
                mu_out = np.mean(heatmap_norm[mask_out]) if np.any(mask_out) else 0.0
                mask_in_hotspot = (maschera_coco == 1) & (clean_binary_map == 255)
                
                if num_hotspots == 0 or not np.any(mask_in_hotspot):
                    mask_in = (maschera_coco == 1)
                    mu_in = np.mean(heatmap_norm[mask_in]) if np.any(mask_in) else 0.0
                else:
                    mu_in = np.mean(heatmap_norm[mask_in_hotspot])

                tbr = mu_in / (mu_out + 1e-6)
                
                if tbr <= SOGLIA_TBR and current_failure == "None":
                    current_failure = "FP_LowEnergy"

                # Assegnazione al Tracker Metriche Locali
                if current_failure == "None":
                    # Il modello ha un ground perfetto in questo token, ci fermiamo!
                    best_metrics = {
                        'target_token': target_token_text, 'token_index': t_idx,
                        'num_target_hotspots': num_hotspots, 'num_distractors': num_distractors,
                        'kmeans_threshold': soglia_kmeans_norm, 'tbr': tbr
                    }
                    image_tp_found = True
                    image_fallback_used = used_fallback
                    image_failure_reason = "None"
                    print(f"      [+] Token Perfetto! (TBR: {tbr:.2f}) - Grounding completato.")
                    break
                else:
                    # Se fallisce, salviamo comunque i dati del token con il TBR più alto per il CSV
                    if tbr > best_tbr_tracker or best_metrics['token_index'] == -1:
                        best_tbr_tracker = tbr
                        best_metrics = {
                            'target_token': target_token_text, 'token_index': t_idx,
                            'num_target_hotspots': num_hotspots, 'num_distractors': num_distractors,
                            'kmeans_threshold': soglia_kmeans_norm, 'tbr': tbr
                        }
                        image_failure_reason = current_failure
                        image_fallback_used = used_fallback
                        
            # --- FINE LOOP TOKEN ---
            if image_tp_found:
                is_correct_id = 1
                lista_tbr_positive.append(best_metrics['tbr'])
                if image_fallback_used: Pos_TP_Fallback += 1
                else: Pos_TP += 1
                print(f"   [🏆] Immagine classificata come True Positive!")
            else:
                if image_failure_reason == "FP_Scattered": Pos_FP_Scattered += 1
                elif image_failure_reason == "FP_Mispointed": Pos_FP_Mispointed += 1
                elif image_failure_reason == "FP_LowEnergy": Pos_FP_LowEnergy += 1
                print(f"   [!] Immagine fallita. Motivo finale: {image_failure_reason}")

    # =========================================================
    # ANALISI NEGATIVA (Allucinazione)
    # =========================================================
    elif query_type == "Negative":
        if ha_negato or not ha_confermato:
            print("   [+] TN: Il modello ha respinto l'oggetto falso.")
            Neg_TN += 1; image_failure_reason = "None"; is_correct_id = 1
        else:
            print("   [!] FP_Alluc: Il modello ha confermato l'oggetto assente!")
            Neg_FP_Alluc += 1; image_failure_reason = "FP_Hallucination"
            
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            
            # Per le negative estraiamo il K-Means sul primissimo token incriminato
            t_idx = target_indices[0] if target_indices else (len(token_ids) - 1)
            target_token_text = processor.decode([token_ids[t_idx]]).strip()
            
            percorso_tam = os.path.join(img_output_dir, f"1_heatmap_tam_alluc_tok{t_idx}.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, t_idx, [], False)

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
            cv2.imwrite(os.path.join(img_output_dir, f"2_kmeans_raw_tok{t_idx}.png"), binary_map_raw)
            
            min_val_in_core = np.min(pixel_values[labels == core_cluster_idx])
            soglia_kmeans_norm = float(min_val_in_core) / 255.0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary_map_morph = cv2.morphologyEx(binary_map_raw, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(binary_map_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_HOTSPOT_AREA]
            
            clean_binary_map = np.zeros_like(binary_map_morph)
            cv2.drawContours(clean_binary_map, valid_contours, -1, 255, -1)
            cv2.imwrite(os.path.join(img_output_dir, f"3_hotspots_clean_tok{t_idx}.png"), clean_binary_map) 
            
            best_metrics = {
                'target_token': target_token_text, 'token_index': t_idx,
                'num_target_hotspots': 0, 'num_distractors': len(valid_contours),
                'kmeans_threshold': soglia_kmeans_norm, 'tbr': 0.0
            }

    # ---------------------------------------------------------
    # SALVATAGGIO ALLINEATO CON GLI ALTRI SCRIPT
    # ---------------------------------------------------------
    risultati.append({
        'img_id': img_id, 'query_type': query_type, 'prompt': prompt.replace('\n', ' | '), 
        'output': testo_generato.replace('\n', ' '), 
        'target_token': best_metrics['target_token'],
        'token_index': best_metrics['token_index'],
        'num_target_hotspots': best_metrics['num_target_hotspots'],
        'num_distractors': best_metrics['num_distractors'],
        'kmeans_threshold': round(best_metrics['kmeans_threshold'], 4),
        'tbr': round(best_metrics['tbr'], 4),
        'is_correct_id': is_correct_id,
        'failure_reason': image_failure_reason,
        'tokens_analyzed': str(target_indices)
    })

    del inputs, image_inputs, video_inputs, outputs, generated_ids
    for var in ['logits', 'heatmap', 'heatmap_raw', 'heatmap_norm', 'heatmap_8bit', 'binary_map_raw', 'binary_map_morph', 'clean_binary_map', 'maschera_coco', 'orig_img', 'hsv_map', 'hue', 'heatmap_isolated', 'main_hotspot_mask', 'debug_vis', 'gt_contours', 'c_mask', 'target_hotspots', 'distractor_hotspots', 'pixel_values', 'labels', 'centers']:
        if var in locals(): del locals()[var]
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(risultati).to_csv(results_file, index=False)

# --- REPORT FINALE ---
print("\n" + "="*60)
print("🎯 REPORT ESPERIMENTO: K-MEANS + CoT SPAZIALE")
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

nome_prompt = "CoT_Spatial_Pipeline"
shutil.make_archive(os.path.join(os.path.dirname(os.path.normpath(output_dir)), f"Test_{nome_prompt}"), 'zip', output_dir)
print(f"\n[SUCCESSO] Archiviato! Controlla la cartella 'visualizations' nell'output.")