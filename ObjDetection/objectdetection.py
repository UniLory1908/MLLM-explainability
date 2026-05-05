import os
import argparse # <-- NUOVO IMPORT per la CLI
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
parser = argparse.ArgumentParser(description="Esegui test TAM e VQA su un sottoinsieme di immagini COCO.")
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
output_dir = './output/test/'

tam_dir = os.path.join(output_dir, 'tam_heatmaps')
os.makedirs(tam_dir, exist_ok=True)

results_file = os.path.join(output_dir, 'results_phase3_vqa.csv') 

print("Inizializzo COCO...")
coco = COCO(ann_file)
df = pd.read_csv(csv_path)

# --- APPLICAZIONE DEL LIMITE CLI ---
if args.num_images is not None:
    # Prende solo le prime 'n' righe dal dataframe
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

# --- VARIABILI PER METRICHE GLOBALI DEL DATASET ---
TP = 0  
FP = 0  
FN = 0  
TN = 0  

lista_obj_iou = []
lista_func_iou = []
lista_f1_iou = []
lista_pixel_acc = []

SOGLIA_IOU_CORRETTO = 0.3

# --- CICLO SULLE IMMAGINI ---
for index, row in df.iterrows():
    img_id = row['img_id']
    img_path = os.path.join(output_dir, row['path'])
    obj_main = row['obj_main']
    ann_id_main = row['ann_id_main']
    
    print(f"\n[{index+1}/{len(df)}] Analizzo immagine: {row['path']} (Oggetto: {obj_main})")
    
    prompt = f"Is there a {obj_main}?"
    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        
    generated_ids = outputs.sequences
    testo_generato = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"   -> Risposta del modello: {testo_generato}")
    
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
    token_ids_generati = generated_ids[0][inputs['input_ids'].shape[1]:].cpu().tolist()
    
    target_idx = -1
    target_token_text = ""
    is_positive_answer = False
    
    testo_pulito = testo_generato.lower()
    if "no" in testo_pulito and "yes" not in testo_pulito:
        print("   [!] Il modello non ha identificato l'oggetto (Risposta Negativa).")
        FN += 1
        risultati.append({
            'img_id': img_id, 'prompt': prompt, 'output': testo_generato, 'target_token': 'NONE',
            'obj_iou': 0.0, 'func_iou': 0.0, 'f1_iou': 0.0, 'pixel_accuracy': 0.0, 'is_correct_id': 0
        })
        df_temp = pd.DataFrame(risultati)
        df_temp.to_csv(results_file, index=False)
        continue 

    is_positive_answer = True
    for i, t_id in enumerate(token_ids_generati):
        parola_token = processor.decode([t_id]).strip().lower()
        parola_pulita = "".join(c for c in parola_token if c.isalpha())
        if obj_main.lower() in parola_pulita and len(parola_pulita) > 1:
            target_idx = i
            target_token_text = parola_token
            break
            
    if target_idx == -1:
        target_idx = 0
        target_token_text = processor.decode([token_ids_generati[0]]).strip()

    # --- ESTRAZIONE MASCHERA COCO ---
    ann = coco.loadAnns(ann_id_main)[0]
    maschera_coco = coco.annToMask(ann) 

    # --- CREAZIONE IMMAGINE ORIGINALE CON MASCHERA SOVRAPPOSTA ---
    img_orig = cv2.imread(img_path)
    if img_orig is not None:
        if maschera_coco.shape != img_orig.shape[:2]:
            maschera_resized = cv2.resize(maschera_coco.astype(np.uint8), (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            maschera_resized = maschera_coco.astype(np.uint8)
        
        contours, _ = cv2.findContours(maschera_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay = img_orig.copy()
        overlay[maschera_resized == 1] = [0, 255, 0] 
        img_with_mask = cv2.addWeighted(img_orig, 0.6, overlay, 0.4, 0)
        cv2.drawContours(img_with_mask, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"gt_mask_{row['path']}"), img_with_mask)

    # --- CALCOLO E SALVATAGGIO TAM ---
    vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
    percorso_salvataggio = os.path.join(tam_dir, row['path'])
    
    heatmap = TAM(generated_ids[0].cpu().tolist(), vision_shape, logits, special_ids, image_inputs, processor, percorso_salvataggio, target_idx, [], False)
    
    if len(heatmap.shape) == 3:
        heatmap_grigia = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        heatmap_norm = heatmap_grigia.astype(np.float32) / 255.0
    else:
        heatmap_norm = heatmap.astype(np.float32)

    # Calcolo Metriche
    obj_iou, func_iou, f1_iou = calcola_metriche_tam(heatmap_norm, maschera_coco)
    pixel_acc = calcola_pixel_accuracy(heatmap_norm, maschera_coco) 
    
    lista_obj_iou.append(obj_iou)
    lista_func_iou.append(func_iou)
    lista_f1_iou.append(f1_iou)
    lista_pixel_acc.append(pixel_acc)

    is_correct_id = 0
    if obj_iou >= SOGLIA_IOU_CORRETTO:
        TP += 1
        is_correct_id = 1
        print(f"   [+] Identificazione Corretta! (Obj-IoU={obj_iou:.3f} >= {SOGLIA_IOU_CORRETTO})")
    else:
        FP += 1
        print(f"   [-] Localizzazione Errata. (Obj-IoU={obj_iou:.3f} < {SOGLIA_IOU_CORRETTO})")

    print(f"   -> Metriche: Obj-IoU={obj_iou:.3f}, Pixel-Acc={pixel_acc:.3f}")

    risultati.append({
        'img_id': img_id, 'prompt': prompt, 'output': testo_generato.replace('\n', ' '), 'target_token': target_token_text,
        'obj_iou': round(obj_iou, 4), 'func_iou': round(func_iou, 4), 'f1_iou': round(f1_iou, 4), 
        'pixel_accuracy': round(pixel_acc, 4), 'is_correct_id': is_correct_id
    })

    del inputs, outputs, logits, generated_ids, heatmap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df_temp = pd.DataFrame(risultati)
    df_temp.to_csv(results_file, index=False)

# --- CALCOLO METRICHE GLOBALI E STATISTICHE FINALI ---
print("\n" + "="*50)
print("🎯 REPORT METRICHE GLOBALI DEL DATASET")
print("="*50)

Accuracy_global = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1_global = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

media_obj_iou = np.mean(lista_obj_iou) if lista_obj_iou else 0
media_func_iou = np.mean(lista_func_iou) if lista_func_iou else 0
media_f1_iou = np.mean(lista_f1_iou) if lista_f1_iou else 0
media_pixel_acc = np.mean(lista_pixel_acc) if lista_pixel_acc else 0

dati_statistici = [
    {'img_id': '---', 'prompt': '--- STATISTICHE FINALI ---', 'output': '---', 'target_token': '---', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Totale Immagini Elaborate', 'prompt': str(len(df)), 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Media Obj-IoU', 'prompt': f"{media_obj_iou:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Media Func-IoU', 'prompt': f"{media_func_iou:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Media F1-IoU', 'prompt': f"{media_f1_iou:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Media Pixel Accuracy', 'prompt': f"{media_pixel_acc:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Veri Positivi (TP)', 'prompt': str(TP), 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Falsi Positivi (FP)', 'prompt': str(FP), 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Falsi Negativi (FN)', 'prompt': str(FN), 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Accuracy Globale', 'prompt': f"{Accuracy_global:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Precision', 'prompt': f"{Precision:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'Recall', 'prompt': f"{Recall:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''},
    {'img_id': 'F1-Score Globale', 'prompt': f"{F1_global:.4f}", 'output': '', 'target_token': '', 'obj_iou': '', 'func_iou': '', 'f1_iou': '', 'pixel_accuracy': '', 'is_correct_id': ''}
]

risultati_completi = risultati + dati_statistici
df_finale = pd.DataFrame(risultati_completi)
df_finale.to_csv(results_file, index=False)

# --- CREAZIONE DEL FILE ZIP ---
nome_prompt = "Is_there_a_item"
# Aggiungiamo anche il numero di immagini al nome dello zip se è stato usato il limite
suffisso_img = f"_{len(df)}img" if args.num_images else "_Full"
nome_zip = f"Test_{nome_prompt}_soglia_{SOGLIA_IOU_CORRETTO}{suffisso_img}"

cartella_padre = os.path.dirname(os.path.normpath(output_dir))
percorso_zip_completo = os.path.join(cartella_padre, nome_zip)

print(f"📦 Sto impacchettando i risultati e le immagini in: {nome_zip}.zip ...")
shutil.make_archive(percorso_zip_completo, 'zip', output_dir)

print("="*50)
print(f"[SUCCESSO] L'archivio {nome_zip}.zip è pronto!")
print("="*50 + "\n")