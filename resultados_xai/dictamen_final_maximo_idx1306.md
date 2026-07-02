# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_maximo_idx1306_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asignó a este cliente una **probabilidad de incumplimiento del 96,1%**, muy por encima del umbral de clasificación del 59,0%. En consecuencia, la **clase predicha es "default" (1)**. Sin embargo, el resultado real observado fue **"no default" (0)**, por lo que nos encontramos ante un **error de predicción (falso positivo)**. El modelo se equivocó con alta confianza. Cabe señalar que el rendimiento global del modelo en el conjunto de prueba es moderado: AUC de 0,773 y F1 de 0,533.

**2. Referencia Base**

El valor base del modelo (*shap_expected_value*) es **0,3038**, que representa la probabilidad promedio de incumplimiento que el modelo asigna a un cliente "típico" de la población de entrenamiento antes de considerar sus características individuales. La tasa de default real en la población de entrenamiento es del **22,12%**. La probabilidad predicha para este cliente (96,1%) se sitúa **66 puntos porcentuales por encima del valor base** y **74 puntos por encima de la tasa poblacional**, indicando un perfil que el modelo considera de riesgo extremo.

**3. Principales Factores de Riesgo**

Todos los cinco factores de mayor impacto en la predicción de este cliente tienen un efecto **positivo** (empujan la probabilidad al alza). No se identificaron factores protectores significativos entre los principales impulsores.

- **PAY_0 (Estado de pago del mes más reciente):** El cliente presenta un valor de **3** (3 meses de atraso en el pago). Esta categoría es **extremadamente rara** en la población de entrenamiento (solo el 1,02% de los clientes). El modelo asocia este nivel de morosidad reciente con un incremento de **+0,159** en la probabilidad de default, siendo el factor individual más influyente.

- **PAY_3 (Estado de pago del mes 3):** Valor de **2** (2 meses de atraso). Esta categoría aparece en el **12,64%** de la población (percentil superior al 75). Contribuye con **+0,0962** a la probabilidad.

- **PAY_5 (Estado de pago del mes 5):** Valor de **7** (7 meses de atraso). Esta categoría es **extremadamente atípica** (solo el 0,17% de los clientes). Aporta **+0,0931** a la probabilidad. Este valor extremo es inusual incluso para el modelo, ya que PAY_5 no es un factor globalmente relevante.

- **PAY_2 (Estado de pago del mes 2):** Valor de **2** (2 meses de atraso). Categoría presente en el **13,02%** de la población. Contribuye con **+0,0881**.

- **LIMIT_BAL (Límite de crédito):** El cliente tiene un límite de **50.000 NT$**, que se sitúa en el **percentil 25** de la población (bajo). El modelo asocia un límite bajo con mayor riesgo, incrementando la probabilidad en **+0,0523**.

**4. Principales Factores Protectores**

Ninguno de los cinco factores de mayor impacto presenta un valor SHAP negativo. Todos ellos empujan la predicción hacia el default. Esto refleja un perfil de riesgo extremo donde el modelo no encuentra señales compensatorias significativas entre las variables más influyentes.

**5. Contexto Global del Modelo**

A nivel global, los factores más importantes del modelo son **PAY_0** (importancia media absoluta: 0,0701) y **LIMIT_BAL** (0,0557), seguidos de PAY_AMT1 y BILL_AMT1. En este cliente, PAY_0 y LIMIT_BAL también figuran entre los principales impulsores, lo que es consistente con el comportamiento general del modelo. Sin embargo, **PAY_5** (importancia global: 0,0068, puesto 14) tiene un impacto local desproporcionadamente alto debido al valor extremo de 7 meses de atraso. Esto indica que, aunque el perfil sigue la lógica general del modelo (la morosidad reciente es el factor dominante), la presencia de un atraso tan prolongado en un mes intermedio genera una señal de riesgo inusualmente fuerte para este cliente en particular.

**6. Nota Metodológica**

Los valores SHAP presentados reflejan **asociaciones aprendidas por el modelo** entre las características del cliente y la probabilidad de default predicha. No deben interpretarse como relaciones causales demostradas. Este análisis explica **qué ponderó el modelo** para llegar a su predicción, no por qué el cliente realmente incumplió o no. La decisión final sobre el cliente debe complementarse con juicio experto y no basarse únicamente en la puntuación del modelo.