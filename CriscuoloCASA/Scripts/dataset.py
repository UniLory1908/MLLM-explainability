import pandas as pd
from pycocotools.coco import COCO
import os


ann_file = 'C:\\Users\\lucac\\Desktop\\TAM-main\\dataset\\annotations\\instances_val2017.json' 

if not os.path.exists(ann_file):
    print(f"Error")
    exit()


coco = COCO(ann_file)

img_ids = coco.getImgIds()

subset_data = []
target_count = 100 
counter = 0

for img_id in img_ids:
    counter += 1
    
    if len(subset_data) >= target_count:
        break
        
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    
    # Filtro 1
    if len(anns) > 4:
        continue
        
    # Filtro 2
    valid_anns = [a for a in anns if 'segmentation' in a and a.get('area', 0) > 5000]
    
    # Filtro 3
    if 1 <= len(valid_anns) <= 2:
        img_info = coco.loadImgs(img_id)[0]
        row = {'img_id': img_id, 'path': img_info['file_name']}
        
        for i, ann in enumerate(valid_anns):
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if i == 0:
                row['obj_main'] = cat_name
                row['ann_id_main'] = ann['id']
            else:
                row[f'obj_{i}'] = cat_name
                row[f'ann_id_{i}'] = ann['id']
                
        subset_data.append(row)

df = pd.DataFrame(subset_data)
df.to_csv('dataset.csv', index=False)
print(f"Done")