# -*- coding: utf-8 -*-
"""
Evaluación Modular de Modelos Base (Baselines) para Riesgo Crediticio
Carga los artefactos exportados por el pipeline principal para garantizar
exactamente el mismo split 70/15/15 y las mismas transformaciones.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

ARTEFACTOS_PKL = "data_artefacts.pkl"

print(f"\n--- CARGANDO ARTEFACTOS DESDE {ARTEFACTOS_PKL} ---")
try:
    with open(ARTEFACTOS_PKL, "rb") as f:
        artefactos = pickle.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ARTEFACTOS_PKL}. Ejecuta primero el pipeline principal.")
    exit()

# 1. RECUPERAR LOS DATOS
X_train_num = artefactos["X_train_num"]
X_train_cat = artefactos["X_train_cat"]
X_val_num   = artefactos["X_val_num"]
X_val_cat   = artefactos["X_val_cat"]
X_test_num  = artefactos["X_test_num"]
X_test_cat  = artefactos["X_test_cat"]

y_train = artefactos["y_train"]
y_val   = artefactos["y_val"]
y_test  = artefactos["y_test"]

# 2. FUSIÓN DE FEATURES PARA MODELOS CLÁSICOS
# Los algoritmos clásicos no usan embeddings separados, reciben una matriz plana
X_train_base = np.hstack((X_train_num, X_train_cat))
X_val_base   = np.hstack((X_val_num, X_val_cat))
X_test_base  = np.hstack((X_test_num, X_test_cat))

# Cálculo del ratio de desbalance para XGBoost y LightGBM
ratio_desbalance = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"Ratio de desbalance calculado: {ratio_desbalance:.2f}:1")

# 3. DEFINICIÓN DE MODELOS
modelos = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    ),
    "SVM (RBF)": SVC(
        class_weight='balanced', 
        probability=True, 
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, 
        scale_pos_weight=ratio_desbalance, 
        random_state=42, 
        eval_metric='auc',
        n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, 
        scale_pos_weight=ratio_desbalance, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
}

resultados_baselines = []

print("\n--- INICIANDO ENTRENAMIENTO Y EVALUACIÓN ---")
# 4. BUCLE DE ENTRENAMIENTO Y OPTIMIZACIÓN DE UMBRAL
for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    
    # Entrenamiento
    modelo.fit(X_train_base, y_train)
    
    # Predicción de probabilidades
    if hasattr(modelo, "predict_proba"):
        probs_val = modelo.predict_proba(X_val_base)[:, 1]
        probs_test = modelo.predict_proba(X_test_base)[:, 1]
    else:
        probs_val = modelo.decision_function(X_val_base)
        probs_test = modelo.decision_function(X_test_base)
        # Normalización min-max para algoritmos sin probabilidad directa
        probs_val = (probs_val - probs_val.min()) / (probs_val.max() - probs_val.min())
        probs_test = (probs_test - probs_test.min()) / (probs_test.max() - probs_test.min())

    # Optimización del Umbral (Threshold) en Validación para maximizar F1-Score
    mejor_f1_val = 0
    mejor_umbral = 0.5
    umbrales = np.arange(0.1, 0.9, 0.01)
    
    for umbral in umbrales:
        preds_val = (probs_val >= umbral).astype(int)
        f1_actual = f1_score(y_val, preds_val)
        if f1_actual > mejor_f1_val:
            mejor_f1_val = f1_actual
            mejor_umbral = umbral
            
    # Evaluación Final en Test usando el umbral óptimo
    preds_test = (probs_test >= mejor_umbral).astype(int)
    
    # Cálculo de métricas
    auc_test = roc_auc_score(y_test, probs_test)
    f1_test = f1_score(y_test, preds_test)
    prec_test = precision_score(y_test, preds_test)
    rec_test = recall_score(y_test, preds_test)
    acc_test = accuracy_score(y_test, preds_test)
    
    # Guardar resultados
    resultados_baselines.append({
        "Modelo": nombre,
        "AUC-ROC": auc_test,
        "F1-Score": f1_test,
        "Precision": prec_test,
        "Recall": rec_test,
        "Accuracy": acc_test,
        "Umbral Óptimo": mejor_umbral
    })
    
    print(f"Completado. Umbral: {mejor_umbral:.2f} | AUC: {auc_test:.4f} | F1: {f1_test:.4f}")

# 5. EXPORTAR RESULTADOS A DATAFRAME
df_resultados = pd.DataFrame(resultados_baselines)

# Mostrar tabla final formateada para facilitar su paso a LaTeX
print("\n=== TABLA DE RESULTADOS PARA LATEX (CONJUNTO DE PRUEBA) ===")
print(df_resultados[["Modelo", "AUC-ROC", "F1-Score", "Precision", "Recall"]].round(4).to_string(index=False))

# Guardar en CSV para tener el respaldo
df_resultados.to_csv("resultados_baselines.csv", index=False)
print("\nResultados exportados a 'resultados_baselines.csv'")