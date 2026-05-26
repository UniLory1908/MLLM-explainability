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

parser = argparse.ArgumentParser(description="VQA Misto (75% Positivi, 25% Negativi) con Hard Prompt.")
parser.add_argument('-n', '--num_images', type=int, default=None)
args = parser.parse_args()

csv_path = './dataset.csv'
ann_file = './dataset/annotations/instances_val2017.json'
output_dir = './output/test_mixed/'
tam_dir = os.path.join(output_dir, 'tam_heatmaps')
os.makedirs(tam_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'results_mixed_hard.csv') 

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
SOGLIA_IOU_CORRETTO = 0.3
risultati = []

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
    else:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cat_presenti = set([a['category_id'] for a in anns])
        cat_assenti = [c for c in tutte_le_categorie.keys() if c not in cat_presenti]
        target_obj = tutte_le_categorie[random.choice(cat_assenti)]
        query_type = "Negative"
        
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
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.0, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    testo_pulito = testo_generato.lower()
    
    ha_confermato = "yes" in testo_pulito or (target_obj.lower() in testo_pulito and len(target_obj) > 1)
    ha_negato = "no" in testo_pulito and "yes" not in testo_pulito
    
    # Valori di default OMOLOGATI per il CSV
    is_correct_id = 0
    obj_iou, func_iou, f1_iou, pixel_acc = 0.0, 0.0, 0.0, 0.0
    target_token_text = "NONE"
    target_idx = -1
    
    if query_type == "Positive":
        if ha_negato or not ha_confermato:
            print("   [-] FN: Non ha visto l'oggetto.")
            # target_token rimane "NONE", is_correct_id rimane 0
        else:
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            maschera_coco = coco.annToMask(coco.loadAnns(ann_id_main)[0]) 
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_pos_step{target_idx}.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            
            heatmap_norm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if len(heatmap.shape) == 3 else heatmap.astype(np.float32)
            obj_iou, func_iou, f1_iou = calcola_metriche_tam(heatmap_norm, maschera_coco)
            pixel_acc = calcola_pixel_accuracy(heatmap_norm, maschera_coco)
            
            if obj_iou >= SOGLIA_IOU_CORRETTO:
                print(f"   [+] TP! (IoU: {obj_iou:.3f})")
                is_correct_id = 1
            else:
                print(f"   [!] FP_loc (Errore Posizione). (IoU: {obj_iou:.3f})")
                
    elif query_type == "Negative":
        if ha_negato or not ha_confermato:
            print("   [+] TN: Resiste all'allucinazione.")
            is_correct_id = 1 
            # target_token rimane "NONE" -> corretto!
        else:
            print("   [!] FP_alluc: Allucinazione!")
            logits = [model.lm_head(f[-1]) for f in outputs.hidden_states]
            token_ids = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
            for i, t_id in enumerate(token_ids):
                parola = processor.decode([t_id]).strip().lower()
                if "yes" in parola or target_obj.lower() in parola:
                    target_idx = i; target_token_text = parola; break
            if target_idx == -1: target_idx = 0; target_token_text = processor.decode([token_ids[0]]).strip()
            
            vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
            percorso_tam = os.path.join(tam_dir, f"{row['path']}_hallucination_step{target_idx}.jpg")
            heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_tam, target_idx, [], False)
            # Niente calcolo IoU perché è un'allucinazione (tutto a 0.0)

    # Salvataggio riga CSV omologata
    risultati.append({
        'img_id': img_id, 'query_type': query_type, 'prompt': prompt, 
        'output': testo_generato.replace('\n', ' '), 'target_token': target_token_text, 
        'token_index': target_idx, 'obj_iou': round(obj_iou, 4), 'func_iou': round(func_iou, 4), 
        'f1_iou': round(f1_iou, 4), 'pixel_accuracy': round(pixel_acc, 4), 'is_correct_id': is_correct_id
    })

    # PULIZIA ESTREMA CPU (Evita contaminazioni tra immagini)
    del inputs, image_inputs, video_inputs, outputs, generated_ids
    if 'logits' in locals(): del logits
    if 'heatmap' in locals(): del heatmap
    if 'maschera_coco' in locals(): del maschera_coco
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(risultati).to_csv(results_file, index=False)

print("\n📦 Test Hard Prompt Mixed concluso e CSV salvato!")