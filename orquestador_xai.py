# -*- coding: utf-8 -*-
"""
Tesis: Arquitectura Híbrida FT-Transformer + LLM para Riesgo Crediticio
Módulo: Orquestador Local con API de DeepSeek (Procesamiento en Lotes)
"""

import os
import json
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. CARGA DE CONFIGURACIÓN LOCAL
# ==========================================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEP_SEEK_KEY')

if not DEEPSEEK_API_KEY:
    raise ValueError("Error: No se encontró la API Key (DEEPSEEK_API_KEY) en el entorno.")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ==========================================
# 2. DEFINICIÓN DE TEMPLATES (PROMPTS)
# ==========================================
# Definiremos estos en detalle más adelante.
PROMPT_EXPLAINER_TEMPLATE = """## Role
You are an expert Credit Risk Explanation Analyst. Your task is to produce a clear, faithful, and technically rigorous verbal explanation of why a credit default risk model (FT-Transformer) assigned a specific default probability to a particular client, using SHAP attribution values as interpretability evidence.

You are NOT a financial advisor. You do NOT make recommendations. You do NOT judge the client as a person. You interpret model outputs only.

## Available Information
You will receive three input blocks:

1. **[CONTEXT]** — `contexto_explicacion.json`: Contains the dataset schema, semantic description of each column, population statistics computed on the training set (mean, std, percentiles per numeric feature; distribution per categorical feature), the model's test metrics (AUC, F1), the classification threshold, the SHAP expected_value (baseline probability), and a methodological note on the nature of SHAP attributions.

2. **[SHAP GLOBAL]** — `shap_global.json`: Contains the global feature importance ranking of the model (mean absolute SHAP and signed mean SHAP per feature), computed over the full test set. Use this to contextualize whether the features that dominate THIS client's explanation also tend to dominate the model globally, or whether this client has an atypical attribution pattern.

3. **[CLIENT]** — `cliente_riesgo_*_xai.json`: Contains one specific client's data:
   - `perfil_riesgo`: risk tier label (Minimo / Bajo / Medio / Alto / Maximo)
   - `probabilidad_default`: the model's predicted probability of default
   - `threshold_clasificacion`: the optimal classification threshold (F1-maximized on validation)
   - `clase_predicha`: the model's binary decision (0=no default, 1=default)
   - `clase_real`: the actual observed outcome for this client
   - `perfil_cliente_real`: the client's raw, unscaled feature values with human-readable categorical labels
   - `inputs_modelo_estandarizados`: the standardized/encoded inputs actually fed to the model
   - `shap_local`: full SHAP attribution vector for this client, sorted by |shap_value| descending
   - `top_5_factores_impacto`: the five features with the largest absolute SHAP impact

## Core Constraint: SHAP Faithfulness
SHAP values are correlational attributions within the model, NOT causal explanations.
- A positive SHAP value means that feature PUSHED the predicted probability ABOVE the baseline (`shap_expected_value`).
- A negative SHAP value means that feature PULLED the predicted probability BELOW the baseline.
- The following must hold approximately: `shap_expected_value + sum(shap_local.values()) ≈ probabilidad_default`
- Never state that a feature "causes" the client to default. Use language like: "the model associates this value with...", "this feature contributed... to the predicted risk", "this factor pushed/pulled the prediction by...".

## Output Format
Your response must contain two clearly delimited sections:

### Section 1 — Chain of Thought (inside <think></think> tags)
This section is mandatory and must be non-empty. It is your private reasoning workspace — the Verifier will audit it directly. Follow these steps in order:

**Step 1 — Verify the arithmetic anchor**
State the `shap_expected_value` (baseline probability for this model/dataset). 
Sum the top-5 SHAP values from `top_5_factores_impacto` and compare to (`probabilidad_default` − `shap_expected_value`).
Note whether the top-5 explain most of the deviation from baseline, or whether the remaining features also contribute materially.

**Step 2 — Classify each top-5 feature**
For each feature in `top_5_factores_impacto`, record:
- The SHAP value and its sign (risk-increasing if positive, risk-reducing if negative)
- The client's raw value from `perfil_cliente_real`
- How that raw value compares to the population (use `estadisticas_numericas` for numeric features: is the value below p25, between p25-p50, between p50-p75, or above p75? For categorical features, use `distribucion_categoricas_pct` to note how common that category is).
- For features with non-trivial codes (PAY_0 through PAY_6), decode the value using the semantic description in `descripcion_columnas`.

**Step 3 — Check global vs. local alignment**
Compare the client's top-5 features against `importancia_global_mean_abs_shap` from `shap_global.json`.
Note which of the client's top drivers are also globally important (typical pattern) and which are atypical (locally dominant for this client, but minor globally). This tells you whether this client's risk profile follows the model's general logic or has idiosyncratic drivers.

**Step 4 — Decision boundary reasoning**
State `probabilidad_default`, `threshold_clasificacion`, and `clase_predicha` explicitly.
Compute the margin: how far is the predicted probability from the threshold? A small margin means borderline; a large margin means clear-cut.
Note whether `clase_predicha` matches `clase_real` (correct prediction) or not (error). If the model erred, flag this explicitly — the explanation is of the model's prediction, not the ground truth.

**Step 5 — Draft the explanation outline**
Briefly list what you will say in the final explanation: which features to highlight, what population comparisons to make, how to frame the risk direction, and what disclaimer to include. Confirm no feature's direction is reversed relative to its SHAP sign.

### Section 2 — Final Explanation (after </think>)
CRITICAL REQUIREMENT: The entire explanation in this section MUST be written STRICTLY IN FORMAL SPANISH. 
Write a clear, structured explanation in plain language (no raw JSON). The audience is a technically literate analyst or risk committee — not the client themselves. Structure it using these translated Spanish headers:

**1. Resumen de la Predicción (Prediction Summary)**
State the model's predicted default probability, the classification threshold, the resulting decision (predicted class), and — if available — whether the prediction matched the actual outcome. Include the model's test-set AUC and F1 as a caveat on overall reliability.

**2. Referencia Base (Baseline Reference)**
State the SHAP expected value and what it represents (the average predicted probability across the population the model was calibrated on, i.e., the starting point before any individual feature adjustments). Compare `probabilidad_default` to both the baseline and the population default rate from `tasa_base_default_pct`.

**3. Principales Factores de Riesgo (Key Risk Drivers: positive SHAP — pushed probability up)**
For each feature with a positive SHAP value among the top-5:
- Name the feature and give its semantic meaning (from `descripcion_columnas`).
- State the client's value in human-readable form (from `perfil_cliente_real`).
- Contextualize it against the population (e.g., "above the 75th percentile", "the most common category in the training set").
- State the SHAP contribution and what it means: "this pushed the model's prediction X points above baseline."
- Use association language, not causal language.

**4. Principales Factores Protectores (Key Protective Factors: negative SHAP — pulled probability down)**
Same structure as above, for features with negative SHAP values among the top-5.
If no top-5 feature has a negative SHAP value (edge case for extreme risk clients), note this explicitly.

**5. Contexto Global del Modelo (Global Model Context)**
Briefly note whether the features driving this client's prediction are also the model's globally most important features, or whether this client has an unusual attribution pattern that deviates from the model's typical logic.

**6. Nota Metodológica (Methodological Disclaimer)**
Close with a short disclaimer: these attributions reflect the model's internal learned associations, not proven causal relationships. The model's decisions should be complemented by human expert judgment, and individual clients should not be reduced to their predicted score.

---

## [CONTEXT]
{CONTEXTO_EXPLICACION_JSON}

## [SHAP GLOBAL]
{SHAP_GLOBAL_JSON}

## [CLIENT]
{CLIENTE_JSON}"""

PROMPT_VERIFIER_TEMPLATE = """## Role
You are a Verifier model specialized in evaluating the faithfulness, logical soundness, and factual consistency of LLM-generated explanations of SHAP attributions for a credit default risk model (FT-Transformer trained on the Taiwan Credit Card Default dataset).
You must carefully inspect the raw SHAP/context data, the reasoning process (CoT), and the final explanation to ensure all three components are aligned and accurate. You MUST answer strictly according to the "How-to-reply" guide.

## Guidelines
- You must act as an evaluator-verifier: do not produce new explanations, do not rewrite the Explainer's output, do not give advice to the client.
- All judgments must be supported by explicit evidence found in the input sections (Context, SHAP Data, CoT, Explanation). Never use external knowledge about credit scoring to "fill gaps" the data does not support.
- The XAI method used is always SHAP (KernelExplainer). There is no need to branch by method.
- Note: The Explanation will be in Spanish. Evaluate the semantic meaning and factual accuracy of the Spanish text against the Context and SHAP Data without penalizing for the language difference.
- Distinguish between **global** explanations (about the model's overall behavior) and **local** explanations (about one specific client). Some rules below apply differently depending on scope.

## Problem Statement
You are tasked with evaluating a verbal explanation produced by an LLM (the "Explainer") that attempts to interpret SHAP attributions for either (a) the global behavior of the credit default model, or (b) one specific client's predicted default probability. Your goal is to determine whether the explanation is faithful, logically consistent, numerically accurate, and non-hallucinated relative to the provided SHAP data and dataset context.

## Solution Structure
1. Confirm the CoT section (<think></think> block) is present and non-empty. If missing or empty, flag as **MissingCoT**.
2. Verify that the CoT reasoning aligns factually with the SHAP values: every numeric SHAP value mentioned or implied in the CoT must match the provided data (within rounding tolerance).
3. Ensure the logical progression from the CoT to the final Explanation is sound — the Explanation must not introduce conclusions, rankings, or numbers that don't trace back to the CoT.
4. **Top-feature ranking check (SwapTopFeature / SwapMinorFeature):**
   - For **local** explanations: the features presented as the top drivers of risk/protection must match the top entries of `shap_local` sorted by `abs(shap_value)` (or `top_5_factores_impacto`). Flag **SwapTopFeature** if a feature outside the true top-ranked set is presented as a top driver, or if the order of clearly dominant features is materially misrepresented.
   - For **global** explanations: same check against `importancia_global_mean_abs_shap`.
   - Flag **SwapMinorFeature** for the same issue when it occurs among lower-ranked/minor features instead of the top ones.
5. **Directional consistency (NegateRelation):** For every feature discussed, the stated direction ("increases risk" / "decreases risk" / "protective factor") must match the sign of its SHAP value (`shap_local[feature] > 0` → increases predicted default probability; `< 0` → decreases it). Also check that statements about raw feature magnitude (e.g., "a high LIMIT_BAL") are not used to imply a SHAP direction opposite to the one actually computed. Flag **NegateRelation** on any mismatch.
6. **Omission check (OmitFeature):**
   - For **global** explanations: all features in the top-K (e.g., top 5) referenced as "what the Explainer was asked to cover" must appear; flag missing ones as **OmitFeature**.
   - For **local** explanations: omitting minor features is acceptable; only flag **OmitFeature** if the Explainer was explicitly asked to cover the full `top_5_factores_impacto` and skipped one of those without justification.
7. **Quantitative grounding (FabricatedValue):** Any numeric value the Explanation attributes to the client (feature values, percentiles, "above/below average" claims) must match the raw data and population statistics exactly (within rounding). Flag **FabricatedValue** if the Explanation invents a number, percentile, population comparison, or statistic not present in the context or client JSON.
8. **Threshold/decision consistency (ThresholdMismatch):** The stated risk category, predicted class, or decision language must be consistent with `probabilidad_default` vs. `threshold_clasificacion` and with `clase_predicha`. Flag **ThresholdMismatch** if the Explanation's framing of the decision contradicts these fields.
9. **Hallucination check (InsertHallucination):** Flag if the Explanation introduces:
   - Causal claims presented as proven fact rather than model associations (e.g., "this client defaults *because* of X" instead of "the model associates X with higher predicted risk").
   - Claims about real-world creditworthiness, fraud, or client intent that go beyond what the model and SHAP values support.
   - Contradictions between the stated mechanism and the actual SHAP data.
   - Statements that the model is "certain," "always right," or omit/misstate metrics when those are referenced.
   Do **not** flag as hallucination if the Explanation uses reasonable, appropriately-hedged language consistent with the SHAP values.
10. Evaluate clarity and impartiality: the Explanation should avoid emotionally charged, stigmatizing, or discriminatory language about the client.
11. If multiple issues are found, report the single most severe one in `error_type` (severity order: MissingCoT > FabricatedValue > NegateRelation > SwapTopFeature > ThresholdMismatch > InsertHallucination > OmitFeature > SwapMinorFeature > None), and mention the others in `justification`.

### How to reply
Respond ONLY with the requested JSON structure. Do not include markdown formatting, backticks, or any introductory text.
Assign a confidence score between 0.0 and 1.0. Deduct 0.1 for every minor ambiguity found in the explanation that did not warrant a full rejection.

{{
    "confidence": float between 0 and 1,
    "verdict": "accept" or "reject",
    "justification": "brief, explicit reasoning referencing evidence from the Context, SHAP Data, CoT, and Explanation sections",
    "error_type": "SwapTopFeature" | "SwapMinorFeature" | "NegateRelation" | "OmitFeature" | "MissingCoT" | "FabricatedValue" | "ThresholdMismatch" | "InsertHallucination" | "None"
}}

## Context & SHAP Data

{CONTEXT_AND_DATA}

## CoT Evaluated
{COT}

## Explanation Evaluated
{EXPLANATION}"""

# ==========================================
# 3. FUNCIONES OPERATIVAS Y FEEDBACK LOOP
# ==========================================

def extraer_cot_y_explicacion(texto_llm):
    """Separa el bloque de razonamiento de la explicación usando Regex."""
    cot_match = re.search(r'<think>(.*?)</think>', texto_llm, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""
    explicacion = re.sub(r'<think>.*?</think>', '', texto_llm, flags=re.DOTALL).strip()
    return cot, explicacion

def llamar_explainer_deepseek(prompt):
    """Inferencia para generar el dictamen técnico."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a specialized AI assistant for Explainable AI (XAI)."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def llamar_verifier_deepseek(prompt):
    """Inferencia con forzado JSON para auditar la veracidad matemática."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a strict verification system. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def orquestador_xai_deepseek(contexto_str, global_str, cliente_str, max_intentos=3):
    """Bucle de retroalimentación Explainer-Verifier con guardado de logs."""
    prompt_actual = PROMPT_EXPLAINER_TEMPLATE.format(
        CONTEXTO_EXPLICACION_JSON=contexto_str,
        SHAP_GLOBAL_JSON=global_str,
        CLIENTE_JSON=cliente_str
    )
    
    datos_combinados = f"[CONTEXT]:\n{contexto_str}\n\n[CLIENT DATA]:\n{cliente_str}"
    
    # Rastro de auditoría completo para este cliente
    historial_intentos = []
    
    for intento in range(1, max_intentos + 1):
        print(f"  -> Intento {intento}/{max_intentos}...")
        
        # Estructura del log para el intento actual
        registro_intento = {
            "intento_numero": intento,
            "cot_generado": "",
            "explicacion_generada": "",
            "veredicto_verifier": "",
            "error_type": "",
            "justificacion_verifier": "",
            "feedback_inyectado": False
        }
        
        texto_explainer = llamar_explainer_deepseek(prompt_actual)
        cot, explicacion = extraer_cot_y_explicacion(texto_explainer)
        
        registro_intento["cot_generado"] = cot
        registro_intento["explicacion_generada"] = explicacion
        
        if not cot:
            print("     ⚠️ Error: Bloque <think> ausente.")
            motivo_rechazo = "MissingCoT: You did not include the <think> tags. Try again."
            registro_intento["error_type"] = "MissingCoT"
            registro_intento["veredicto_verifier"] = "reject"
        else:
            prompt_verifier = PROMPT_VERIFIER_TEMPLATE.format(
                CONTEXT_AND_DATA=datos_combinados,
                COT=cot,
                EXPLANATION=explicacion
            )
            
            texto_verifier = llamar_verifier_deepseek(prompt_verifier)
            
            try:
                evaluacion = json.loads(texto_verifier)
                veredicto = evaluacion.get("verdict", "reject").lower()
                error = evaluacion.get("error_type", "None")
                justificacion = evaluacion.get("justification", "")
                
                registro_intento["veredicto_verifier"] = veredicto
                registro_intento["error_type"] = error
                registro_intento["justificacion_verifier"] = justificacion
                
                if veredicto == "accept":
                    print("     ✅ Certificado exitosamente.")
                    historial_intentos.append(registro_intento)
                    return {
                        "estado": "exito",
                        "intentos_totales": intento,
                        "explicacion_final": explicacion,
                        "historial_completo": historial_intentos
                    }
                else:
                    print(f"     ❌ Rechazado: {error} - {justificacion}")
                    motivo_rechazo = f"Error Type: {error}. Justification: {justificacion}"
                    
            except json.JSONDecodeError:
                print("     ⚠️ Error: La API no retornó JSON válido.")
                motivo_rechazo = "FormatError: Return strictly valid JSON."
                registro_intento["error_type"] = "FormatError"
                registro_intento["veredicto_verifier"] = "reject"

        if intento < max_intentos:
            registro_intento["feedback_inyectado"] = True
            prompt_actual += f"\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\nYour previous explanation was REJECTED by the Verifier.\nReason for rejection: {motivo_rechazo}\nPlease generate a new response correcting this specific error. Ensure your new <think> section explicitly addresses this mistake before writing the Final Explanation in Spanish.\n"
            time.sleep(1.5) # Pausa para evitar rate-limits
            
        historial_intentos.append(registro_intento)
            
    print("  ❌ Se agotaron los intentos.")
    return {
        "estado": "fallo",
        "intentos_totales": max_intentos,
        "historial_completo": historial_intentos
    }

# ==========================================
# 4. PUNTO DE ENTRADA INDUSTRIAL (BATCH PROCESSING)
# ==========================================
if __name__ == "__main__":
    archivo_contexto = "contexto_explicacion.json"
    archivo_global = "shap_global.json"
    
    archivos_clientes = [
        "cliente_riesgo_minimo_idx715_xai.json",
        "cliente_riesgo_bajo_idx635_xai.json",
        "cliente_riesgo_medio_idx4095_xai.json",
        "cliente_riesgo_alto_idx318_xai.json",
        "cliente_riesgo_maximo_idx1306_xai.json"
    ]
    
    # Crear directorios para organizar salidas
    os.makedirs("resultados_xai", exist_ok=True)
    os.makedirs("logs_auditoria", exist_ok=True)
    
    print("Cargando contexto global del modelo...")
    with open(archivo_contexto, "r", encoding="utf-8") as f:
        ctx_str = f.read()
    with open(archivo_global, "r", encoding="utf-8") as f:
        glb_str = f.read()

    print("\n--- INICIANDO PROCESAMIENTO XAI POR LOTES ---")
    
    for archivo_cliente in archivos_clientes:
        print(f"\nProcesando perfil: {archivo_cliente}")
        
        with open(archivo_cliente, "r", encoding="utf-8") as f:
            cli_str = f.read()

        resultado = orquestador_xai_deepseek(ctx_str, glb_str, cli_str, max_intentos=3)
        perfil_str = archivo_cliente.replace("cliente_riesgo_", "").replace("_xai.json", "")
        
        # Guardar el log de auditoría completo siempre (incluso si falla)
        archivo_log = f"logs_auditoria/audit_log_{perfil_str}.json"
        with open(archivo_log, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=4, ensure_ascii=False)
        print(f"  💾 Log de auditoría guardado en: {archivo_log}")
        
        # Guardar el dictamen final solo si fue certificado exitosamente
        if resultado["estado"] == "exito":
            archivo_reporte = f"resultados_xai/dictamen_final_{perfil_str}.md"
            with open(archivo_reporte, "w", encoding="utf-8") as f:
                f.write(f"# DICTAMEN TÉCNICO DE EXPLICABILIDAD\n")
                f.write(f"**Archivo Origen:** {archivo_cliente}\n")
                f.write(f"**Intentos de Certificación:** {resultado['intentos_totales']}/3\n")
                f.write(f"---\n\n")
                f.write(resultado["explicacion_final"])
            print(f"  📄 Reporte final exportado a: {archivo_reporte}")
        else:
            print(f"  ⚠️ No se generó dictamen final para {perfil_str} debido a fallos de certificación.")