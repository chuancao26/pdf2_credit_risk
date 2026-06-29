# -*- coding: utf-8 -*-
"""
Pipeline End-to-End: Adquisición de datos + EDA + Entrenamiento + SHAP
para FT-Transformer sobre el dataset Taiwan Credit Card Default (UCI id=350).

Hardware objetivo: GTX 1050 Ti (4GB VRAM)

Este script fusiona en uno solo lo que antes eran dos pasos separados:
  1) Descarga, limpieza, EDA, encoding, split y normalización de los datos.
  2) Construcción, entrenamiento y auditoría XAI (SHAP) del FT-Transformer.

Ya no es necesario guardar 'data_artefacts.pkl' en Colab y bajarlo a la PC:
todo corre en el mismo proceso. Aun así, se guarda un .pkl de respaldo por
si quieres reutilizar los artefactos sin volver a descargar el dataset.
"""

import os
import sys
import gc
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import rtdl_revisiting_models as rtdl
import shap

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

# Rutas de salida (ajusta si quieres separar resultados de cada corrida)
ARTEFACTOS_PKL = "data_artefacts.pkl"
MODEL_PATH = "ft_transformer_best.pt"

# ==========================================
# 0. CONFIGURACIÓN DE HARDWARE
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--- Iniciando Sistema ---")
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. ADQUISICIÓN DE DATOS
# ==========================================
print("\nDescargando dataset desde UCI ML Repository...")
dataset = fetch_ucirepo(id=350)  # ID oficial del Taiwan Credit Card Default

X_raw = dataset.data.features.copy()
y_raw = dataset.data.targets.copy()

COLUMN_MAPPING = {
    "X1":  "LIMIT_BAL",   # Monto de crédito otorgado
    "X2":  "SEX",          # Género
    "X3":  "EDUCATION",    # Nivel educativo
    "X4":  "MARRIAGE",     # Estado civil
    "X5":  "AGE",          # Edad
    "X6":  "PAY_0",        # Estado de pago mes 1 (septiembre 2005)
    "X7":  "PAY_2",        # Estado de pago mes 2 (agosto 2005)
    "X8":  "PAY_3",        # Estado de pago mes 3
    "X9":  "PAY_4",        # Estado de pago mes 4
    "X10": "PAY_5",        # Estado de pago mes 5
    "X11": "PAY_6",        # Estado de pago mes 6 (abril 2005)
    "X12": "BILL_AMT1",    # Monto factura mes 1
    "X13": "BILL_AMT2",    # Monto factura mes 2
    "X14": "BILL_AMT3",    # Monto factura mes 3
    "X15": "BILL_AMT4",    # Monto factura mes 4
    "X16": "BILL_AMT5",    # Monto factura mes 5
    "X17": "BILL_AMT6",    # Monto factura mes 6
    "X18": "PAY_AMT1",     # Monto pagado mes 1
    "X19": "PAY_AMT2",     # Monto pagado mes 2
    "X20": "PAY_AMT3",     # Monto pagado mes 3
    "X21": "PAY_AMT4",     # Monto pagado mes 4
    "X22": "PAY_AMT5",     # Monto pagado mes 5
    "X23": "PAY_AMT6",     # Monto pagado mes 6
}

X_raw = X_raw.rename(columns=COLUMN_MAPPING)
print(list(X_raw.columns))

# NUMÉRICAS:
#   LIMIT_BAL   - Monto de crédito otorgado (NT$)
#   AGE         - Edad del cliente (años)
#   BILL_AMT1-6 - Monto de factura meses 1 al 6
#   PAY_AMT1-6  - Monto de pago realizado meses 1 al 6
#
# CATEGÓRICAS (requieren embedding en FT-Transformer):
#   SEX         - 1=Masculino, 2=Femenino
#   EDUCATION   - 1=Posgrado, 2=Universidad, 3=Secundaria, 4=Otro
#   MARRIAGE    - 1=Casado, 2=Soltero, 3=Otro
#   PAY_0-6     - Estado de pago mensual (-1=pago puntual, 1-8=meses de retraso)
#
# TARGET:
#   default.payment.next.month - 1=Incumplimiento, 0=Pago correcto

nulos = X_raw.isnull().sum()
print("\nValores nulos por columna:")
print(nulos[nulos > 0] if nulos.sum() > 0 else " No hay")

duplicados = X_raw.duplicated().sum()
print(f"\nFilas duplicadas: {duplicados}")

print("\nTipos de datos:")
print(X_raw.dtypes)

print("\nEstadísticas descriptivas:")
print(X_raw.describe().round(2))

# ==========================================
# 2. LIMPIEZA DE DATOS
# ==========================================
X = X_raw.copy()
y = y_raw.copy()
y.columns = ["target"]

antes = len(X)
X = X.drop_duplicates()
y = y.loc[X.index]
print(f"\nDuplicados eliminados: {antes - len(X)}")

# valores inválidos en EDUCATION (0,5,6 no documentados -> "Otros"=4)
print(f"\nEDUCATION - valores únicos antes: {sorted(X['EDUCATION'].unique())}")
X['EDUCATION'] = X['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
print(f"EDUCATION - valores únicos después: {sorted(X['EDUCATION'].unique())}")

# valores inválidos en MARRIAGE (0 -> "Otro"=3)
print(f"\nMARRIAGE - valores únicos antes: {sorted(X['MARRIAGE'].unique())}")
X['MARRIAGE'] = X['MARRIAGE'].replace({0: 3})
print(f"MARRIAGE - valores únicos después: {sorted(X['MARRIAGE'].unique())}")

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"\nShape final: {X.shape}")

# ==========================================
# 3. EDA
# ==========================================
conteo = y["target"].value_counts()
proporcion = y["target"].value_counts(normalize=True) * 100

print("\nDistribución del target:")
print(f"  Clase 0 (No default): {conteo[0]:,} ({proporcion[0]:.1f}%)")
print(f"  Clase 1 (Default):    {conteo[1]:,} ({proporcion[1]:.1f}%)")
print(f"  Ratio desbalance:     {conteo[0] / conteo[1]:.1f}:1")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(["No default\n(0)", "Default\n(1)"],
            [conteo[0], conteo[1]],
            color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.5)
axes[0].set_title("Distribución del target", fontweight="bold")
axes[0].set_ylabel("Número de clientes")
for i, v in enumerate([conteo[0], conteo[1]]):
    axes[0].text(i, v + 200, f"{v:,}\n({proporcion[i]:.1f}%)", ha="center", fontsize=10)

for clase, color, label in [(0, "#4C72B0", "No default"), (1, "#DD8452", "Default")]:
    mask = y["target"] == clase
    axes[1].hist(X.loc[mask, "AGE"], bins=30, alpha=0.6,
                 color=color, label=label, edgecolor="white", linewidth=0.3)
axes[1].set_title("Distribución de edad por clase", fontweight="bold")
axes[1].set_xlabel("Edad")
axes[1].legend()

edu_labels = {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otro"}
edu_default = pd.concat([X["EDUCATION"], y["target"]], axis=1).groupby("EDUCATION")["target"].mean() * 100
edu_default.index = [edu_labels.get(i, str(i)) for i in edu_default.index]
axes[2].bar(edu_default.index, edu_default.values, color="#55A868", edgecolor="white")
axes[2].set_title("Tasa de default por educación (%)", fontweight="bold")
axes[2].set_ylabel("% default")
axes[2].set_ylim(0, 35)
for i, v in enumerate(edu_default.values):
    axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("--> Gráfico EDA guardado como '01_eda_overview.png'")

# ==========================================
# 4. COLUMNAS POR TIPO
# ==========================================
CAT_COLS = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]

NUM_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

print(f"Columnas categóricas ({len(CAT_COLS)}): {CAT_COLS}")
print(f"Columnas numéricas   ({len(NUM_COLS)}): {NUM_COLS}")

# Diccionario de referencia para que el LLM explicador entienda los códigos
# categóricos del "perfil real del cliente" (no son los índices de LabelEncoder,
# son los valores originales del dataset, ya limpiados).
DICCIONARIO_CATEGORIAS = {
    "SEX": {"1": "Masculino", "2": "Femenino"},
    "EDUCATION": {"1": "Posgrado", "2": "Universidad", "3": "Secundaria", "4": "Otro"},
    "MARRIAGE": {"1": "Casado", "2": "Soltero", "3": "Otro"},
    "PAY_0_a_PAY_6": {
        "-2": "Sin consumo / sin línea usada",
        "-1": "Pago puntual",
        "0": "Pago mínimo dentro del plazo (uso de crédito revolvente)",
        "1": "1 mes de retraso",
        "2": "2 meses de retraso",
        "...": "N meses de retraso (hasta 8)",
    },
}

# ==========================================
# 5. ENCODING DE VARIABLES CATEGÓRICAS
# ==========================================
X_encoded = X.copy()
label_encoders = {}
cat_cardinalities = []

for col in CAT_COLS:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le
    n_cats = len(le.classes_)
    cat_cardinalities.append(n_cats)
    print(f"  {col:12s} → {n_cats} categorías: {list(le.classes_)}")

print(f"\nCardinalidades (para FT-Transformer): {cat_cardinalities}")

# ==========================================
# 6. DIVISIÓN DE LOS DATOS (70/15/15)
# ==========================================
X_temp, X_test, y_temp, y_test_s = train_test_split(
    X_encoded, y["target"],
    test_size=0.15, stratify=y["target"], random_state=42
)

X_train, X_val, y_train_s, y_val_s = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85, stratify=y_temp, random_state=42
)

# Conservamos el perfil REAL (sin label-encoding ni estandarización) de los splits,
# alineado posicionalmente con los arrays usados por el modelo. X_test_raw_real se usa
# para mostrar el "perfil del cliente" en unidades originales; X_train_raw_real se usa
# para calcular estadísticas poblacionales de referencia (contexto para el explainer LLM).
X_test_raw_real = X.loc[X_test.index].reset_index(drop=True)
X_train_raw_real = X.loc[X_train.index].reset_index(drop=True)

print(f"Train:      {len(X_train):,} filas  ({len(X_train) / len(X) * 100:.1f}%)")
print(f"Validation: {len(X_val):,}  filas  ({len(X_val) / len(X) * 100:.1f}%)")
print(f"Test:       {len(X_test):,}  filas  ({len(X_test) / len(X) * 100:.1f}%)")

for nombre, y_split in [("Train", y_train_s), ("Val", y_val_s), ("Test", y_test_s)]:
    pct = y_split.mean() * 100
    print(f"  {nombre} - % default: {pct:.1f}%")

# Perfil REAL del cliente: valores originales, SIN estandarizar (NUM_COLS en su
# escala natural) y SIN label-encoding (CAT_COLS con sus códigos originales del
# dataset, ej. SEX=1/2, EDUCATION=1-4, PAY_0 en su rango real -2..8).
# Se usa 'X' (post-limpieza, pre-encoding/pre-escalado), no los arrays que
# alimentan al modelo. El orden de filas queda alineado con X_test (y por lo
# tanto con X_test_num / X_test_cat / test_combined_global más abajo).
X_test_real = X.loc[X_test.index].reset_index(drop=True)

# ==========================================
# 7. NORMALIZACIÓN Y PESOS DE CLASE
# ==========================================
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[NUM_COLS]).astype(np.float32)
X_val_num   = scaler.transform(X_val[NUM_COLS]).astype(np.float32)
X_test_num  = scaler.transform(X_test[NUM_COLS]).astype(np.float32)

X_train_cat = X_train[CAT_COLS].values.astype(np.int64)
X_val_cat   = X_val[CAT_COLS].values.astype(np.int64)
X_test_cat  = X_test[CAT_COLS].values.astype(np.int64)

y_train = y_train_s.values.astype(np.float32)
y_val   = y_val_s.values.astype(np.float32)
y_test  = y_test_s.values.astype(np.float32)

for col, mean, std in zip(NUM_COLS[:5], scaler.mean_[:5], scaler.scale_[:5]):
    print(f"  {col:12s}: media={mean:,.0f}, std={std:,.0f}")

clases = np.array([0, 1])
pesos = compute_class_weight("balanced", classes=clases, y=y_train_s.values)
class_weights = dict(zip(clases, pesos))

print("\nPesos de clase para pérdida balanceada:")
print(f"  Clase 0 (No default): {class_weights[0]:.4f}")
print(f"  Clase 1 (Default):    {class_weights[1]:.4f}")
print(f"  → El modelo penalizará {class_weights[1] / class_weights[0]:.1f}x más los errores en defaulters")

N_NUM_FEATURES = len(NUM_COLS)
CAT_CARDINALITIES = cat_cardinalities
CLASS_WEIGHTS = class_weights
feature_names_num = NUM_COLS
feature_names_cat = CAT_COLS
todas_las_caracteristicas = feature_names_num + feature_names_cat

print(f"\n  X_train_num shape: {X_train_num.shape}")
print(f"  X_train_cat shape: {X_train_cat.shape}")
print(f"  cat_cardinalities: {CAT_CARDINALITIES}")
print(f"  n_num_features:    {N_NUM_FEATURES}")

print("\nRESUMEN PARA PRESENTACIÓN DE AVANCES")
print(f"  Dataset:               Taiwan Credit Card Default (UCI id=350)")
print(f"  Observaciones totales: {len(X):,}")
print(f"  Features numéricas:    {len(NUM_COLS)}")
print(f"  Features categóricas:  {len(CAT_COLS)}")
print(f"  Split Train/Val/Test:  70% / 15% / 15% (estratificado)")
print(f"  Desbalance de clases:  {y['target'].mean() * 100:.1f}% defaulters")
print(f"  Manejo desbalance:     class_weight balanceado")
print(f"  Encoding categórico:   LabelEncoder (índices para FT-Transformer)")
print(f"  Normalización num.:    StandardScaler (fit solo en train)")

# ==========================================
# 8. RESPALDO DE ARTEFACTOS (opcional)
# ==========================================
artefactos = {
    "X_train_num": X_train_num, "X_train_cat": X_train_cat,
    "X_val_num": X_val_num, "X_val_cat": X_val_cat,
    "X_test_num": X_test_num, "X_test_cat": X_test_cat,
    "y_train": y_train, "y_val": y_val, "y_test": y_test,
    "cat_cardinalities": CAT_CARDINALITIES,
    "n_num_features": N_NUM_FEATURES,
    "cat_cols": CAT_COLS, "num_cols": NUM_COLS,
    "scaler": scaler, "label_encoders": label_encoders,
    "class_weights": CLASS_WEIGHTS,
    "X_test_raw": X_test.reset_index(drop=True),
    "y_test_raw": y_test_s.reset_index(drop=True),
}
with open(ARTEFACTOS_PKL, "wb") as f:
    pickle.dump(artefactos, f)
print(f"\n--> Respaldo de artefactos guardado en '{ARTEFACTOS_PKL}' (opcional, no se vuelve a leer en esta corrida)")

# ==========================================
# 9. DATALOADERS
# ==========================================
BATCH_SIZE = 256  # Tamaño de batch seguro para 4GB VRAM en entrenamiento

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_num), torch.tensor(X_train_cat), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_num), torch.tensor(X_val_cat), torch.tensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test_num), torch.tensor(X_test_cat), torch.tensor(y_test)),
    batch_size=BATCH_SIZE, shuffle=False,
)

# ==========================================
# 10. CONSTRUCCIÓN DEL MODELO
# ==========================================
defaults = rtdl.FTTransformer.get_default_kwargs()
model = rtdl.FTTransformer(
    n_cont_features=N_NUM_FEATURES,
    cat_cardinalities=CAT_CARDINALITIES,
    **defaults,
    d_out=1,
).to(DEVICE)

# ==========================================
# 11. ENTRENAMIENTO (si no existe el modelo guardado)
# ==========================================
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
# 11.5 THRESHOLD ÓPTIMO DE CLASIFICACIÓN
# ==========================================
# El FT-Transformer entrega probabilidades; para convertirlas en clase 0/1 se
# necesita un umbral. Aquí se busca el umbral que maximiza el F1 en el set de
# validación (más robusto que usar 0.5 a secas dado el desbalance de clases).
print("\n--- Calculando threshold óptimo de clasificación (max F1 en validación) ---")
model.eval()
val_probs_final, val_labels_final = [], []
with torch.no_grad():
    for x_num, x_cat, y_batch in val_loader:
        logits = model(x_num.to(DEVICE), x_cat.to(DEVICE)).squeeze(1)
        probs = torch.sigmoid(logits)
        val_probs_final.extend(probs.cpu().numpy())
        val_labels_final.extend(y_batch.numpy())

val_probs_final = np.array(val_probs_final)
val_labels_final = np.array(val_labels_final)

thresholds_grid = np.arange(0.05, 0.96, 0.01)
f1_scores_grid = [f1_score(val_labels_final, (val_probs_final >= t).astype(int)) for t in thresholds_grid]
THRESHOLD = float(thresholds_grid[int(np.argmax(f1_scores_grid))])
print(f"--> Threshold óptimo seleccionado: {THRESHOLD:.2f} (F1 val = {max(f1_scores_grid):.4f})")

# ==========================================
# 12. EXPLICABILIDAD (SHAP) - OPTIMIZADO
# ==========================================
print("\n--- Iniciando Auditoría XAI (SHAP) ---")
model.eval()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

background_size = 200
np.random.seed(42)
bg_indices = np.random.choice(len(X_train_num), background_size, replace=False)
bg_combined = np.hstack([X_train_num[bg_indices], X_train_cat[bg_indices]])
print("Calculando clusters base...")
bg_summary = shap.kmeans(bg_combined, 20)


def predict_wrapper_batched(X_combined):
    batch_size = 1024  # Batch hiper-seguro para 4GB VRAM
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(all_probs)


explainer = shap.KernelExplainer(predict_wrapper_batched, bg_summary)

# Valor base (expected_value): la probabilidad promedio de referencia desde la cual
# cada SHAP value suma/resta para llegar a la predicción final del cliente.
SHAP_EXPECTED_VALUE = float(np.array(explainer.expected_value).flatten()[0])
print(f"--> SHAP expected_value (valor base): {SHAP_EXPECTED_VALUE:.4f}")

test_size_global = len(X_test_num)
test_combined_global = np.hstack([X_test_num[:test_size_global], X_test_cat[:test_size_global]])

print(f"Calculando SHAP Values para {test_size_global} muestras. Esto puede tomar unos minutos...")
shap_values_global = explainer.shap_values(test_combined_global, nsamples=150)

print("\nGuardando Gráfico Global...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_global, test_combined_global, feature_names=todas_las_caracteristicas,
                   plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("importancia_global.png", dpi=300)
plt.close()
print("--> Gráfico guardado como 'importancia_global.png'")

# --- Probabilidades y métricas del modelo en test (con el threshold óptimo) ---
probabilidades = predict_wrapper_batched(test_combined_global)
predicciones_test = (probabilidades >= THRESHOLD).astype(int)
test_auc = roc_auc_score(y_test[:test_size_global], probabilidades)
test_f1 = f1_score(y_test[:test_size_global], predicciones_test)
print(f"--> AUC Test: {test_auc:.4f} | F1 Test (threshold={THRESHOLD:.2f}): {test_f1:.4f}")

# --- Guardar SHAP global en un JSON aparte ---
print("\nGuardando SHAP global en JSON...")
importancia_global_abs = np.abs(shap_values_global).mean(axis=0)
importancia_global_signed = shap_values_global.mean(axis=0)

importancia_global_dict = dict(zip(todas_las_caracteristicas, [round(float(v), 4) for v in importancia_global_abs]))
importancia_global_dict = dict(sorted(importancia_global_dict.items(), key=lambda item: item[1], reverse=True))

importancia_global_signed_dict = dict(zip(todas_las_caracteristicas, [round(float(v), 4) for v in importancia_global_signed]))
importancia_global_signed_dict = dict(sorted(importancia_global_signed_dict.items(), key=lambda item: item[1], reverse=True))

shap_global_export = {
    "feature_names": todas_las_caracteristicas,
    "shap_expected_value": round(SHAP_EXPECTED_VALUE, 4),
    "n_muestras_evaluadas": int(test_size_global),
    "threshold_clasificacion": THRESHOLD,
    "metricas_modelo_test": {
        "auc": round(float(test_auc), 4),
        "f1": round(float(test_f1), 4),
    },
    "importancia_global_mean_abs_shap": importancia_global_dict,
    "importancia_global_promedio_con_signo": importancia_global_signed_dict,
}

with open("shap_global.json", "w") as f:
    json.dump(shap_global_export, f, indent=4)
print("--> JSON exportado exitosamente: 'shap_global.json'")

# ==========================================
# 13. CONTEXTO PARA EL EXPLAINER (LLM) — ZERO-SHOT + CHAIN OF THOUGHT
# ==========================================
# Este bloque arma todo el "marco de referencia" que un LLM necesita para razonar
# sobre las predicciones del modelo sin alucinar umbrales de "alto/bajo" ni
# relaciones causales que el modelo no garantiza. Se exporta a 'contexto_explicacion.json'
# y se reutiliza igual para los 5 perfiles representativos (no cambia por cliente).
print("\n--- Construyendo bloque de CONTEXTO para el explainer (LLM) ---")

# --- 13.1 Mapeos para mostrar las categóricas en su forma humana/real ---
SEX_LABELS = {1: "Masculino", 2: "Femenino"}
EDUCATION_LABELS = {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otro"}
MARRIAGE_LABELS = {1: "Casado", 2: "Soltero", 3: "Otro"}

# --- 13.2 Descripción semántica de cada columna del dataset ---
DESCRIPCION_COLUMNAS = {
    "LIMIT_BAL": "Monto de crédito otorgado al cliente (NT$, dólar taiwanés). Variable numérica.",
    "AGE": "Edad del cliente, en años. Variable numérica.",
    "SEX": "Género del cliente. Categórica: 1=Masculino, 2=Femenino.",
    "EDUCATION": "Nivel educativo del cliente. Categórica: 1=Posgrado, 2=Universidad, 3=Secundaria, 4=Otro.",
    "MARRIAGE": "Estado civil del cliente. Categórica: 1=Casado, 2=Soltero, 3=Otro.",
}
for i in range(1, 7):
    DESCRIPCION_COLUMNAS[f"BILL_AMT{i}"] = (
        f"Monto de la factura del mes {i} (NT$). Mes 1 = más reciente, mes 6 = más antiguo. Variable numérica."
    )
    DESCRIPCION_COLUMNAS[f"PAY_AMT{i}"] = (
        f"Monto efectivamente pagado por el cliente en el mes {i} (NT$). Mes 1 = más reciente. Variable numérica."
    )
PAY_COL_ORDEN = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
for i, col in enumerate(PAY_COL_ORDEN, start=1):
    DESCRIPCION_COLUMNAS[col] = (
        f"Estado de pago del mes {i} (mes 1 = más reciente; nota: el dataset original no incluye 'PAY_1', "
        f"'{col}' es el nombre real de esa columna en la fuente UCI). Categórica ordinal: "
        "-1 = pago puntual, 0 = pago mínimo/uso de crédito revolvente sin atraso registrado, "
        "1 = 1 mes de atraso, 2 = 2 meses de atraso, ..., 8 = 8 meses o más de atraso."
    )

# --- 13.3 Estadísticas poblacionales de referencia (calculadas SOLO sobre train, valores reales sin escalar) ---
estadisticas_numericas = {}
for col in NUM_COLS:
    serie = X_train_raw_real[col].astype(float)
    estadisticas_numericas[col] = {
        "media": round(float(serie.mean()), 2),
        "std": round(float(serie.std()), 2),
        "min": round(float(serie.min()), 2),
        "p25": round(float(serie.quantile(0.25)), 2),
        "p50_mediana": round(float(serie.quantile(0.50)), 2),
        "p75": round(float(serie.quantile(0.75)), 2),
        "max": round(float(serie.max()), 2),
    }

distribucion_categoricas = {}
for col in CAT_COLS:
    conteo_col = X_train_raw_real[col].value_counts(normalize=True).sort_index() * 100
    if col == "SEX":
        etiquetas = {int(k): SEX_LABELS.get(int(k), int(k)) for k in conteo_col.index}
    elif col == "EDUCATION":
        etiquetas = {int(k): EDUCATION_LABELS.get(int(k), int(k)) for k in conteo_col.index}
    elif col == "MARRIAGE":
        etiquetas = {int(k): MARRIAGE_LABELS.get(int(k), int(k)) for k in conteo_col.index}
    else:
        etiquetas = {int(k): int(k) for k in conteo_col.index}
    distribucion_categoricas[col] = {
        str(etiquetas[int(k)]): round(float(v), 2) for k, v in conteo_col.items()
    }

# --- 13.4 Tasa base de incumplimiento (referencia para "qué tan riesgoso es lo normal") ---
tasa_base_default_train = round(float(y_train_s.mean() * 100), 2)

# --- 13.5 Instrucción explícita sobre la naturaleza de SHAP (anti-causalidad) ---
NOTA_METODOLOGICA_SHAP = (
    "Los valores SHAP son atribuciones de tipo correlacional dentro del modelo entrenado: "
    "indican cuánto desplaza cada variable la predicción del modelo respecto al valor base "
    "(shap_expected_value), NO relaciones de causa-efecto comprobadas en la realidad. "
    "Evita afirmar que una variable 'causa' el incumplimiento; describe que el modelo "
    "'asocia' o 'pondera' esa variable como factor de riesgo o protección. La suma de "
    "shap_expected_value + todos los shap_local de un cliente es igual (aprox.) a la "
    "probabilidad de default predicha para ese cliente."
)

contexto_export = {
    "descripcion_dataset": {
        "nombre": "Taiwan Credit Card Default (UCI Machine Learning Repository, id=350)",
        "objetivo": "Predecir si un cliente incurrirá en incumplimiento de pago (default) el próximo mes.",
        "n_observaciones_total": int(len(X)),
        "columnas_numericas": NUM_COLS,
        "columnas_categoricas": CAT_COLS,
        "descripcion_columnas": DESCRIPCION_COLUMNAS,
    },
    "estadisticas_poblacionales_train": {
        "numericas": estadisticas_numericas,
        "categoricas_distribucion_pct": distribucion_categoricas,
        "tasa_base_default_pct": tasa_base_default_train,
    },
    "modelo": {
        "arquitectura": "FT-Transformer (rtdl_revisiting_models)",
        "metricas_test": {
            "auc": round(float(test_auc), 4),
            "f1": round(float(test_f1), 4),
        },
        "threshold_clasificacion": THRESHOLD,
        "shap_expected_value": round(SHAP_EXPECTED_VALUE, 4),
    },
    "nota_metodologica_shap": NOTA_METODOLOGICA_SHAP,
}

with open("contexto_explicacion.json", "w") as f:
    json.dump(contexto_export, f, indent=4, ensure_ascii=False)
print("--> JSON exportado exitosamente: 'contexto_explicacion.json'")

# ==========================================
# 14. SERIALIZACIÓN (5 PERFILES REPRESENTATIVOS)
# ==========================================
print("\n--- EXTRACCION DE VECTORES LOCALES REPRESENTATIVOS ---")

indices_ordenados_por_riesgo = np.argsort(probabilidades)
n_total = len(indices_ordenados_por_riesgo)

indices_representativos = [
    indices_ordenados_por_riesgo[0],
    indices_ordenados_por_riesgo[int(n_total * 0.25)],
    indices_ordenados_por_riesgo[int(n_total * 0.50)],
    indices_ordenados_por_riesgo[int(n_total * 0.75)],
    indices_ordenados_por_riesgo[-1],
]

etiquetas_riesgo = ["Minimo", "Bajo", "Medio", "Alto", "Maximo"]


def construir_perfil_real(fila_raw):
    """Convierte una fila de X_test_raw_real (sin encoding ni estandarización)
    en un diccionario legible, con las categóricas traducidas a su etiqueta real."""
    perfil = {}
    for col in NUM_COLS:
        perfil[col] = round(float(fila_raw[col]), 2)
    for col in CAT_COLS:
        valor_codigo = int(fila_raw[col])
        if col == "SEX":
            perfil[col] = SEX_LABELS.get(valor_codigo, valor_codigo)
        elif col == "EDUCATION":
            perfil[col] = EDUCATION_LABELS.get(valor_codigo, valor_codigo)
        elif col == "MARRIAGE":
            perfil[col] = MARRIAGE_LABELS.get(valor_codigo, valor_codigo)
        else:
            perfil[col] = valor_codigo
    return perfil


for idx_relativo, perfil in zip(indices_representativos, etiquetas_riesgo):
    impacto_local = shap_values_global[idx_relativo]
    probabilidad_local = probabilidades[idx_relativo]
    clase_predicha_local = int(probabilidad_local >= THRESHOLD)

    # SHAP local: vector completo, redondeado a 4 decimales y ordenado por |impacto|
    vector_local = {feat: round(float(val), 4) for feat, val in zip(todas_las_caracteristicas, impacto_local)}
    vector_local_ordenado = dict(sorted(vector_local.items(), key=lambda item: abs(item[1]), reverse=True))

    print(f"\n[ Perfil de Riesgo: {perfil.upper()} | Cliente Indice: {idx_relativo} ]")
    print(f"Probabilidad de Default: {probabilidad_local:.4f} | Clase predicha (threshold {THRESHOLD:.2f}): {clase_predicha_local}")
    print("Top 5 impactos locales:")
    for feature, impact in list(vector_local_ordenado.items())[:5]:
        print(f"  {feature:25s}: {impact:+.4f}")

    # Perfil REAL del cliente: valores originales (sin LabelEncoder ni StandardScaler)
    perfil_cliente_real = construir_perfil_real(X_test_raw_real.iloc[idx_relativo])

    # Vector exacto que se le dio al modelo (numérico estandarizado + categóricas codificadas),
    # útil para que el explainer pueda cruzar cada valor de SHAP con el input real del modelo.
    inputs_modelo = {feat: round(float(val), 4)
                      for feat, val in zip(todas_las_caracteristicas, test_combined_global[idx_relativo])}

    # Top 5 factores de impacto, como referencia rápida (no es un prompt, es solo un resumen)
    top_5_factores = [
        {"feature": feat, "shap_value": val}
        for feat, val in list(vector_local_ordenado.items())[:5]
    ]

    datos_exportar = {
        "perfil_riesgo": perfil,
        "indice_cliente_test": int(idx_relativo),
        "probabilidad_default": round(float(probabilidad_local), 4),
        "threshold_clasificacion": THRESHOLD,
        "clase_predicha": clase_predicha_local,
        "clase_real": float(y_test[idx_relativo]),
        "perfil_cliente_real": perfil_cliente_real,
        "inputs_modelo_estandarizados": inputs_modelo,
        "shap_local": vector_local_ordenado,
        "top_5_factores_impacto": top_5_factores,
    }

    nombre_archivo_json = f"cliente_riesgo_{perfil.lower()}_idx{idx_relativo}_xai.json"
    with open(nombre_archivo_json, "w") as f:
        json.dump(datos_exportar, f, indent=4, ensure_ascii=False)

    print(f"--> JSON exportado exitosamente: {nombre_archivo_json}")

print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")