# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_maximo_idx1306_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asignó a este cliente una **probabilidad de incumplimiento del 96,1%** (0,961), muy por encima del umbral de clasificación de 0,590. En consecuencia, la **clase predicha fue "default" (1)**. Sin embargo, el resultado real observado fue **"no default" (0)**, por lo que nos encontramos ante un **falso positivo** (error Tipo I). El modelo, en su conjunto, presenta un AUC en test de 0,773 y un F1 de 0,533, lo que indica una capacidad discriminativa moderada pero con margen de error.

**2. Referencia Base**

El valor esperado SHAP (línea base) del modelo es **0,3038**, que representa la probabilidad promedio de incumplimiento en la población sobre la que fue calibrado el modelo. La probabilidad predicha para este cliente (0,961) se sitúa **0,657 puntos por encima de dicha línea base**, y también muy por encima de la tasa de default poblacional del 22,12%. Esto indica que el modelo percibe a este cliente como de riesgo extremadamente alto.

**3. Principales Factores de Riesgo**

Todos los cinco factores de mayor impacto en la predicción de este cliente presentan valores SHAP positivos, lo que significa que **empujaron la probabilidad de default al alza**, alejándola de la línea base. Se detallan a continuación ordenados por magnitud de impacto:

*   **PAY_0 (Estado de pago del mes más reciente):** El cliente presenta un valor de **3** (3 meses de atraso en el pago). Esta categoría es **extremadamente infrecuente** en la población de entrenamiento (solo el 1,02% de los casos). El modelo asocia este severo retraso reciente con un incremento de **+0,159** en la probabilidad de default. Es, con diferencia, el factor más influyente.

*   **PAY_3 (Estado de pago del mes 3):** El cliente presenta un valor de **2** (2 meses de atraso). Esta categoría es menos común pero no excepcional (12,64% de la población). El modelo asocia este atraso con un incremento de **+0,096** en la probabilidad.

*   **PAY_5 (Estado de pago del mes 5):** El cliente presenta un valor de **7** (7 meses de atraso). Esta es una categoría **extremadamente atípica** (solo el 0,17% de la población de entrenamiento). El modelo pondera este severo retraso histórico con un incremento de **+0,093** en la probabilidad.

*   **PAY_2 (Estado de pago del mes 2):** El cliente presenta un valor de **2** (2 meses de atraso). Al igual que PAY_3, esta categoría está presente en el 13,02% de la población. El modelo asocia este atraso con un incremento de **+0,088** en la probabilidad.

*   **LIMIT_BAL (Límite de crédito otorgado):** El cliente dispone de un límite de **50.000 NT$**. Este valor se sitúa en el **percentil 25** de la distribución poblacional (media: 167.780 NT$), lo que indica un límite de crédito bajo. El modelo asocia esta característica con un incremento de **+0,052** en la probabilidad de default.

**4. Principales Factores Protectores**

En el conjunto de los cinco factores de mayor impacto para este cliente, **no se identifica ningún factor protector** (todos los valores SHAP son positivos). Esto es coherente con un perfil de riesgo extremo, donde todas las señales relevantes para el modelo apuntan en la misma dirección de incremento del riesgo.

**5. Contexto Global del Modelo**

Los factores que dominan la explicación local de este cliente (PAY_0, PAY_2, PAY_3 y LIMIT_BAL) coinciden con las variables más importantes a nivel global del modelo FT-Transformer. Esto indica que el perfil de riesgo de este cliente sigue la **lógica general del modelo**. No obstante, destaca la presencia de **PAY_5** como el tercer factor local más influyente, cuando a nivel global ocupa el puesto 14 en importancia. Este peso inusualmente alto se debe al valor extremo de 7 meses de atraso que presenta el cliente, una condición que el modelo aprende a penalizar severamente, aunque sea poco frecuente en la población general.

**6. Nota Metodológica**

Es fundamental señalar que los valores SHAP presentados son **atribuciones correlacionales internas del modelo**, no relaciones de causalidad demostradas. Describen cómo el modelo pondera cada variable para llegar a su predicción, pero no implican que un valor de PAY_0=3 "cause" el incumplimiento. Esta explicación debe ser interpretada por un analista experto y complementada con juicio humano y conocimiento del negocio. El modelo, con un AUC de 0,773, tiene una capacidad predictiva limitada y, como se ha visto en este caso, puede incurrir en errores de clasificación.