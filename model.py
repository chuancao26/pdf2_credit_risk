# -*- coding: utf-8 -*-
"""
Script End-to-End: Entrenamiento + SHAP para FT-Transformer
Hardware Objetivo: GTX 1050 Ti (4GB VRAM)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import rtdl_revisiting_models as rtdl
from sklearn.metrics import roc_auc_score, f1_score
import shap
import gc
import matplotlib.pyplot as plt
import json
import os
import warnings
import sys

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DE HARDWARE Y DATOS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Iniciando Sistema ---")
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# Verificar que el archivo de datos exista
if not os.path.exists("data_artefacts.pkl"):
    print("\n[ERROR] No se encontró 'data_artefacts.pkl'.")
    print("Debes descargar este archivo de tu Google Colab y colocarlo en la misma carpeta que este script.")
    sys.exit(1)

print("\nCargando artefactos de datos...")
with open("data_artefacts.pkl", "rb") as f:
    art = pickle.load(f)

X_train_num = art["X_train_num"].astype(np.float32)
X_train_cat = art["X_train_cat"].astype(np.int64)
X_val_num   = art["X_val_num"].astype(np.float32)
X_val_cat   = art["X_val_cat"].astype(np.int64)
X_test_num  = art["X_test_num"].astype(np.float32)
X_test_cat  = art["X_test_cat"].astype(np.int64)
y_train     = art["y_train"].astype(np.float32)
y_val       = art["y_val"].astype(np.float32)
y_test      = art["y_test"].astype(np.float32)

CAT_CARDINALITIES = art["cat_cardinalities"]
N_NUM_FEATURES    = art["n_num_features"]
CLASS_WEIGHTS     = art["class_weights"]

try:
    feature_names_num = art["num_cols"]
    feature_names_cat = art["cat_cols"]
except KeyError:
    feature_names_num = [f"Num_{i}" for i in range(N_NUM_FEATURES)]
    feature_names_cat = [f"Cat_{i}" for i in range(len(CAT_CARDINALITIES))]

todas_las_caracteristicas = feature_names_num + feature_names_cat

BATCH_SIZE = 256 # Tamaño de batch seguro para 4GB VRAM en entrenamiento
train_loader = DataLoader(TensorDataset(torch.tensor(X_train_num), torch.tensor(X_train_cat), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val_num), torch.tensor(X_val_cat), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_test_num), torch.tensor(X_test_cat), torch.tensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. CONSTRUCCIÓN DEL MODELO
# ==========================================
defaults = rtdl.FTTransformer.get_default_kwargs()
model = rtdl.FTTransformer(
    n_cont_features=N_NUM_FEATURES,
    cat_cardinalities=CAT_CARDINALITIES,
    **defaults,
    d_out=1,
).to(DEVICE)

# ==========================================
# 3. ENTRENAMIENTO (Si no existe el modelo guardado)
# ==========================================
MODEL_PATH = "ft_transformer_best.pt"

if os.path.exists(MODEL_PATH):
    print(f"\n[INFO] Modelo previo detectado ({MODEL_PATH}). Cargando pesos...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("\n--- Iniciando Entrenamiento ---")
    pos_weight = torch.tensor([CLASS_WEIGHTS[1] / CLASS_WEIGHTS[0]], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    N_EPOCHS = 50
    PATIENCE = 10
    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in range(1, N_EPOCHS + 1):
        # TRAIN
        model.train()
        train_losses = []
        for x_num, x_cat, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_num.to(DEVICE), x_cat.to(DEVICE)).squeeze(1)
            loss = criterion(logits, y_batch.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # VALIDATION
        model.eval()
        val_losses, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for x_num, x_cat, y_batch in val_loader:
                logits = model(x_num.to(DEVICE), x_cat.to(DEVICE)).squeeze(1)
                loss = criterion(logits, y_batch.to(DEVICE))
                val_losses.append(loss.item())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y_batch.numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_ctr += 1

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if patience_ctr >= PATIENCE:
            print(f"Early stopping en época {epoch}.")
            break

    print("\nEntrenamiento finalizado. Cargando el mejor modelo para la evaluación...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# ==========================================
# 4. EXPLICABILIDAD (SHAP) - OPTIMIZADO
# ==========================================
print("\n--- Iniciando Auditoría XAI (SHAP) ---")
model.eval()
torch.cuda.empty_cache()
gc.collect()

# Fondo K-Means
background_size = 200
np.random.seed(42)
bg_indices = np.random.choice(len(X_train_num), background_size, replace=False)
bg_combined = np.hstack([X_train_num[bg_indices], X_train_cat[bg_indices]])
print("Calculando clusters base...")
bg_summary = shap.kmeans(bg_combined, 20)

# Wrapper con micro-batches para no quemar los 4GB de VRAM
def predict_wrapper_batched(X_combined):
    batch_size = 1024 # Batch hiper-seguro para 4GB VRAM
    n_samples = X_combined.shape[0]
    all_probs = []

    for i in range(0, n_samples, batch_size):
        batch = X_combined[i:i + batch_size]
        x_n = batch[:, :N_NUM_FEATURES].astype(np.float32)
        x_c = batch[:, N_NUM_FEATURES:].astype(np.int64)

        with torch.no_grad():
            logits = model(torch.tensor(x_n).to(DEVICE), torch.tensor(x_c).to(DEVICE)).squeeze(1)
            probs = torch.sigmoid(logits)
        
        all_probs.extend(probs.cpu().numpy())
        torch.cuda.empty_cache() # Limpiar VRAM en cada ciclo

    return np.array(all_probs)

explainer = shap.KernelExplainer(predict_wrapper_batched, bg_summary)

# Calculamos SHAP solo para 500 muestras del test para que no demore horas en tu PC
test_size_global = len(X_test_num)
test_combined_global = np.hstack([X_test_num[:test_size_global], X_test_cat[:test_size_global]])

print(f"Calculando SHAP Values para {test_size_global} muestras. Esto puede tomar unos minutos...")
shap_values_global = explainer.shap_values(test_combined_global, nsamples=150) # nsamples=150 es un buen balance tiempo/precisión

print("\nGuardando Gráfico Global...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_global, test_combined_global, feature_names=todas_las_caracteristicas, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("importancia_global.png", dpi=300)
plt.close()
print("--> Gráfico guardado como 'importancia_global.png'")

# ==========================================
# 5. SERIALIZACIÓN (5 PERFILES REPRESENTATIVOS)
# ==========================================
print("\n--- EXTRACCION DE VECTORES LOCALES REPRESENTATIVOS ---")

# Calculamos las probabilidades reales de toda la muestra global
probabilidades = predict_wrapper_batched(test_combined_global)

# Encontrar índices representativos basados en los percentiles de riesgo (0%, 25%, 50%, 75%, 100%)
indices_ordenados_por_riesgo = np.argsort(probabilidades)
n_total = len(indices_ordenados_por_riesgo)

indices_representativos = [
    indices_ordenados_por_riesgo[0],                    # Riesgo Mínimo
    indices_ordenados_por_riesgo[int(n_total * 0.25)],  # Riesgo Bajo
    indices_ordenados_por_riesgo[int(n_total * 0.50)],  # Riesgo Medio
    indices_ordenados_por_riesgo[int(n_total * 0.75)],  # Riesgo Alto
    indices_ordenados_por_riesgo[-1]                    # Riesgo Máximo
]

etiquetas_riesgo = ["Minimo", "Bajo", "Medio", "Alto", "Maximo"]

# Procesar y exportar cada uno de los 5 perfiles
for idx_relativo, perfil in zip(indices_representativos, etiquetas_riesgo):
    impacto_local = shap_values_global[idx_relativo]
    probabilidad_local = probabilidades[idx_relativo]

    # Crear el diccionario SHAP local
    vector_local = dict(zip(todas_las_caracteristicas, impacto_local))
    vector_local_ordenado = dict(sorted(vector_local.items(), key=lambda item: abs(item[1]), reverse=True))

    # --- 1. Imprimir en consola ---
    print(f"\n[ Perfil de Riesgo: {perfil.upper()} | Cliente Indice: {idx_relativo} ]")
    print(f"Probabilidad de Default: {probabilidad_local:.4f}")
    print("Top 5 impactos locales:")
    for feature, impact in list(vector_local_ordenado.items())[:5]:
        print(f"  {feature:25s}: {impact:+.4f}")

    # --- 2. Preparar datos para el JSON ---
    perfil_cliente_local = {col: float(test_combined_global[idx_relativo][i]) for i, col in enumerate(todas_las_caracteristicas)}

    # Separar impactos para el prompt
    impactos_positivos = {k: v for k, v in vector_local.items() if v > 0}
    impactos_negativos = {k: v for k, v in vector_local.items() if v < 0}

    top_positivos = dict(sorted(impactos_positivos.items(), key=lambda item: item[1], reverse=True)[:3])
    top_negativos = dict(sorted(impactos_negativos.items(), key=lambda item: item[1])[:3])

    perfil_relevante = {k: perfil_cliente_local.get(k, "N/A") for k in list(top_positivos.keys()) + list(top_negativos.keys())}

    prompt_final = f"""Actua como un auditor experto en riesgos de credito...
[METRICAS DEL MODELO]
- Riesgo de Default calculado: {(probabilidad_local * 100):.2f}%
- Categoria de Riesgo: {perfil.upper()}

[ATRIBUCION DE VARIABLES (SHAP)]
Riesgo (Factores negativos que aumentan la probabilidad): {json.dumps(top_positivos)}
Protección (Factores positivos que reducen la probabilidad): {json.dumps(top_negativos)}

[PERFIL DEL CLIENTE]
{json.dumps(perfil_relevante)}
"""

    datos_exportar = {
        "perfil_riesgo": perfil,
        "probabilidad": float(probabilidad_local),
        "clase_real": float(y_test[idx_relativo]),
        "vector_shap": vector_local,
        "perfil_cliente": perfil_cliente_local,
        "prompt_llm": prompt_final
    }

    # --- 3. Guardar el archivo JSON ---
    nombre_archivo_json = f'cliente_riesgo_{perfil.lower()}_idx{idx_relativo}_xai.json'
    with open(nombre_archivo_json, 'w') as f:
        json.dump(datos_exportar, f, indent=4)

    print(f"--> JSON exportado exitosamente: {nombre_archivo_json}")

print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")