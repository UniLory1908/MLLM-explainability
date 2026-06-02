import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# --- CONFIGURAZIONE ESTETICA PER PUBBLICAZIONE ACCADEMICA ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

output_dir = "grafici_finali"
os.makedirs(output_dir, exist_ok=True)

# --- 1. CARICAMENTO E PREPROCESSING DEI DATI ---
print("Caricamento dataset in corso...")
try:
    df_binario = pd.read_csv('results_mixed_hard_definitivo.csv')
    df_cot = pd.read_csv('results_cot_definitivo.csv')
    df_mcq = pd.read_csv('results_mcq_visual.csv')
except FileNotFoundError as e:
    print(f"Errore: {e}")
    exit()

df_binario['Experiment'] = 'Binary VQA'
df_cot['Experiment'] = 'Spatial CoT'
df_mcq['Experiment'] = 'MCQ'

df_all = pd.concat([df_binario, df_cot, df_mcq], ignore_index=True)
df_all['failure_reason'] = df_all['failure_reason'].fillna('None').astype(str).str.strip()
df_pos = df_all[df_all['query_type'] == 'Positive']

# --- 2. GRAFICO 1: Analisi dei Fallimenti (I 4 Gate) ---
print("Generazione Grafico 1: Categorie di Fallimento...")
df_failures = df_all[df_all['failure_reason'] != 'None']
plt.figure(figsize=(10, 6))
sns.countplot(data=df_failures, x='Experiment', hue='failure_reason', palette='tab10')
plt.title('Ripartizione delle Categorie di Fallimento (XAI Gates)')
plt.ylabel('Frequenza (N. Immagini)')
plt.xlabel('Paradigma Sperimentale')
plt.legend(title='Motivo del Fallimento', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"{output_dir}/1_failure_reasons.png")
plt.close()

# --- 3. GRAFICO 2: Confronto Target-to-Background Ratio (TBR) ---
print("Generazione Grafico 2: Confidenza Visiva (TBR)...")
df_tp = df_all[(df_all['query_type'] == 'Positive') & (df_all['failure_reason'] == 'None')]
if not df_tp.empty:
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df_tp, x='Experiment', y='tbr', palette='muted', inner="quartile")
    sns.swarmplot(data=df_tp, x='Experiment', y='tbr', color="white", edgecolor="gray", linewidth=1, size=6)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Soglia di Accettabilità (TBR=1.0)')
    plt.title('Densità Energetica: Target-to-Background Ratio (Solo True Positives)')
    plt.ylabel('Target-to-Background Ratio (TBR)')
    plt.xlabel('')
    plt.legend()
    plt.savefig(f"{output_dir}/2_tbr_violin.png")
    plt.close()

# --- 4. GRAFICO 3: Analisi Soglia di K-Means ---
print("Generazione Grafico 3: Analisi Soglia K-Means...")
if 'kmeans_threshold' in df_all.columns:
    plt.figure(figsize=(9, 6))
    sns.kdeplot(data=df_pos, x='kmeans_threshold', hue='Experiment', fill=True, common_norm=False, palette='crest', alpha=0.5)
    plt.title('Distribuzione Energetica del Core Focus (Soglia Minima K-Means)')
    plt.xlabel('Intensità Logit Normalizzata (0-1)')
    plt.ylabel('Densità di Probabilità')
    plt.savefig(f"{output_dir}/3_kmeans_threshold.png")
    plt.close()

# --- 5. GRAFICO 4: Gate Survival Cascade (Line Plot) ---
print("Generazione Grafico 4: Gate Survival Cascade...")
survival_data = []
for exp in ['Binary VQA', 'Spatial CoT', 'MCQ']:
    df_exp = df_all[(df_all['Experiment'] == exp) & (df_all['query_type'] == 'Positive')]
    total = len(df_exp)
    pass_gate1 = total - len(df_exp[df_exp['failure_reason'].isin(['FN_Text', 'FN_WrongChoice'])])
    pass_gate3 = pass_gate1 - len(df_exp[df_exp['failure_reason'].isin(['FP_Scattered', 'FP_Mispointed'])])
    pass_gate4 = len(df_exp[df_exp['failure_reason'] == 'None'])
    
    survival_data.append({
        'Experiment': exp,
        '1. Totali (Iniziali)': total,
        '2. Gate Testuale': pass_gate1,
        '3. Gate Geometrico': pass_gate3,
        '4. Gate Energetico (TP)': pass_gate4
    })

df_survival = pd.DataFrame(survival_data).set_index('Experiment').T
df_survival.plot(kind='line', marker='o', linewidth=3, markersize=10, figsize=(10, 6), colormap='Set1')
plt.title('Decadimento delle Performance: Sopravvivenza ai Gate (Query Positive)')
plt.ylabel('Numero di Inferenze Rimanenti')
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Paradigma')
plt.savefig(f"{output_dir}/4_gate_survival_line.png")
plt.close()

# --- 6. GRAFICO 5: Confusion Matrices Affiancate ---
print("Generazione Grafico 5: Matrici di Confusione...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmap = LinearSegmentedColormap.from_list("custom_blue", ["#f7fbff", "#08306b"])

for i, exp in enumerate(['Binary VQA', 'Spatial CoT', 'MCQ']):
    df_exp = df_all[df_all['Experiment'] == exp]
    
    TP = len(df_exp[(df_exp['query_type'] == 'Positive') & (df_exp['failure_reason'] == 'None')])
    FN = len(df_exp[(df_exp['query_type'] == 'Positive') & (df_exp['failure_reason'] != 'None')])
    TN = len(df_exp[(df_exp['query_type'] == 'Negative') & (df_exp['failure_reason'] == 'None')])
    FP = len(df_exp[(df_exp['query_type'] == 'Negative') & (df_exp['failure_reason'] != 'None')])
    
    cm = np.array([[TN, FP], [FN, TP]])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=axes[i], annot_kws={"size": 16},
                xticklabels=['Pred. Negative', 'Pred. Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    axes[i].set_title(f'Confusion Matrix: {exp}', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/5_confusion_matrices.png")
plt.close()

# --- 7. GRAFICO 6: Metriche Globali (Barre Orizzontali) ---
print("Generazione Grafico 6: Metriche Globali (Accuracy, F1, Precision, Recall)...")
metrics_data = []
for exp in ['Binary VQA', 'Spatial CoT', 'MCQ']:
    df_exp = df_all[df_all['Experiment'] == exp]
    
    TP = len(df_exp[(df_exp['query_type'] == 'Positive') & (df_exp['failure_reason'] == 'None')])
    FN = len(df_exp[(df_exp['query_type'] == 'Positive') & (df_exp['failure_reason'] != 'None')])
    TN = len(df_exp[(df_exp['query_type'] == 'Negative') & (df_exp['failure_reason'] == 'None')])
    FP = len(df_exp[(df_exp['query_type'] == 'Negative') & (df_exp['failure_reason'] != 'None')])
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_data.append({'Experiment': exp, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score})

df_metrics = pd.DataFrame(metrics_data)
df_metrics_melted = df_metrics.melt(id_vars='Experiment', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
# Usiamo barre orizzontali, molto più leggibili per confrontare indici percentuali
sns.barplot(data=df_metrics_melted, y='Metric', x='Score', hue='Experiment', palette='viridis')
plt.title('Performance di Visual Grounding (Metriche Classiche)', fontweight='bold')
plt.xlabel('Punteggio (0.0 - 1.0)')
plt.ylabel('')
plt.xlim(0, 1.05)
plt.legend(title='Paradigma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"{output_dir}/6_classical_metrics_horizontal.png")
plt.close()

print(f"\n[SUCCESSO] Archiviato! Controlla la cartella '{output_dir}'.")