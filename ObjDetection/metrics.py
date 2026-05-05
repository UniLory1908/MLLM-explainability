import numpy as np
import cv2

def calcola_iou(maschera_predetta, maschera_reale):
    intersezione = np.logical_and(maschera_predetta, maschera_reale).sum()
    unione = np.logical_or(maschera_predetta, maschera_reale).sum()
    
    if unione == 0:
        return 0.0
    return intersezione / unione

def calcola_metriche_tam(heatmap, maschera_coco):
   
    if heatmap.shape != maschera_coco.shape:
        heatmap = cv2.resize(heatmap, (maschera_coco.shape[1], maschera_coco.shape[0]))

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    soglia_obj = 0.5
    maschera_obj = (heatmap > soglia_obj).astype(np.uint8)
    obj_iou = calcola_iou(maschera_obj, maschera_coco)

    soglia_func = 0.2
    maschera_func = (heatmap > soglia_func).astype(np.uint8)
    func_iou = calcola_iou(maschera_func, maschera_coco)

   
    tp = np.logical_and(maschera_obj, maschera_coco).sum()
    fp = np.logical_and(maschera_obj, np.logical_not(maschera_coco)).sum()
    fn = np.logical_and(np.logical_not(maschera_obj), maschera_coco).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1_iou = 2 * (precision * recall) / (precision + recall)
    else:
        f1_iou = 0.0

    return obj_iou, func_iou, f1_iou

def calcola_pixel_accuracy(heatmap, maschera_coco, soglia=0.5):
    """
    Calcola l'accuratezza pixel per pixel tra la heatmap binarizzata e la maschera COCO.
    Considera corretti sia i pixel dell'oggetto (1=1) sia quelli di sfondo (0=0).
    """
    # --- AGGIUNTA FONDAMENTALE: Ridimensionamento della heatmap ---
    # Se le dimensioni non combaciano, stira la heatmap alla grandezza originale dell'immagine
    if heatmap.shape != maschera_coco.shape:
        heatmap = cv2.resize(heatmap, (maschera_coco.shape[1], maschera_coco.shape[0]))
    
    # Binarizziamo la heatmap
    maschera_predetta = (heatmap > soglia).astype(np.uint8)
    
    # Contiamo i pixel esattamente identici
    pixel_corretti = np.sum(maschera_predetta == maschera_coco)
    pixel_totali = maschera_coco.size
    
    return pixel_corretti / pixel_totali