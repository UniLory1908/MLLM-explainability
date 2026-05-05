import pandas as pd
import os
import urllib.request
import time  # Modulo aggiunto per gestire le pause

output_dir = 'deliverables_images'
os.makedirs(output_dir, exist_ok=True)

csv_path = 'C:\\Users\\lucac\\Desktop\\TAM-main\\dataset.csv'
if not os.path.exists(csv_path):
    print("Errore: Impossibile trovare il file CSV.")
    exit()

df = pd.read_csv(csv_path)
base_url = "http://images.cocodataset.org/val2017/"

print(f"Inizio il controllo e il download di {len(df)} immagini...")

for index, row in df.iterrows():
    filename = row['path']
    img_url = base_url + filename
    save_path = os.path.join(output_dir, filename)
    
    # 1. TRUCCO: Se l'immagine esiste già, saltala!
    if os.path.exists(save_path):
        print(f"[{index+1}/{len(df)}] {filename} già presente. Salto.")
        continue

    print(f"[{index+1}/{len(df)}] Scaricando {filename}...")
    
    # 2. Sistema di tentativi (Retry) in caso di errore di rete
    max_tentativi = 3
    for tentativo in range(max_tentativi):
        try:
            # Scarica il file
            urllib.request.urlretrieve(img_url, save_path)
            print("   -> Successo!")
            
            # 3. Pausa di cortesia per non sovraccaricare il server COCO
            time.sleep(0.5) 
            break  # Se ha successo, esce dal ciclo dei tentativi
            
        except Exception as e:
            # Ora stampiamo la vera variabile 'e' per vedere l'errore reale
            print(f"   -> Errore al tentativo {tentativo + 1}: {e}")
            
            if tentativo < max_tentativi - 1:
                print("   -> Riprovo tra 2 secondi...")
                time.sleep(2)
            else:
                print("   -> [!] Impossibile scaricare l'immagine dopo 3 tentativi.")