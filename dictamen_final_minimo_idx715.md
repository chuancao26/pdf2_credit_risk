# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_minimo_idx715_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una **probabilidad de incumplimiento (default) del 2,69%**, muy por debajo del umbral de clasificación del **59,00%**. En consecuencia, la **clase predicha es 0 (no default)**, lo que coincide con el resultado real observado (clase real = 0). La predicción es correcta y se realiza con un margen muy amplio (56,31 puntos porcentuales por debajo del umbral). Cabe señalar que el modelo presenta un AUC en test de **0,773** y un F1 de **0,533**, lo que indica una capacidad discriminativa moderada pero no perfecta.

**2. Referencia Base**

El valor base de SHAP (*shap_expected_value*) es **0,3038 (30,38%)**, que representa la probabilidad promedio de default que el modelo asigna a la población de entrenamiento antes de considerar las características individuales de cada cliente. La tasa base de default en la población de entrenamiento es del **22,12%**. La probabilidad de este cliente (2,69%) se sitúa muy por debajo de ambos valores de referencia, indicando un perfil de riesgo significativamente menor que el promedio.

**3. Principales Factores de Riesgo (SHAP positivo — empujan la probabilidad al alza)**

- **BILL_AMT3 (Monto de factura del mes 3):** El cliente presenta un saldo de factura de **209.680 NT$** en el tercer mes más reciente. Este valor se encuentra muy por encima del percentil 75 de la población (61.346,75 NT$), lo que representa una facturación inusualmente alta. El modelo asocia este valor con un incremento de la probabilidad de default de **+0,0369** (3,69 puntos porcentuales por encima de la línea base). Es el único factor de riesgo relevante entre los cinco principales, y su efecto es moderado en comparación con los factores protectores.

**4. Principales Factores Protectores (SHAP negativo — empujan la probabilidad a la baja)**

- **PAY_AMT1 (Monto pagado en el mes 1):** El cliente realizó un pago de **100.000 NT$** en el mes más reciente. Este valor es extremadamente alto en comparación con la población (media = 5.682,81 NT$; percentil 75 = 5.006,25 NT$). El modelo asocia este comportamiento con una reducción de la probabilidad de default de **-0,0989** (9,89 puntos porcentuales por debajo de la línea base). Es el factor con mayor impacto absoluto en la predicción.

- **LIMIT_BAL (Límite de crédito):** El cliente dispone de un límite de crédito de **360.000 NT$**, muy por encima del percentil 75 poblacional (240.000 NT$). El modelo asocia un límite elevado con una reducción de la probabilidad de default de **-0,0969** (9,69 puntos porcentuales por debajo de la línea base).

- **PAY_0 (Estado de pago del mes 1):** El cliente presenta un valor de **0** en esta variable, que corresponde a "pago mínimo o uso de crédito revolvente sin atraso registrado" en el mes más reciente. Esta es la categoría más frecuente en la población (49,21%). El modelo asocia esta ausencia de atraso con una reducción de la probabilidad de default de **-0,0755** (7,55 puntos porcentuales por debajo de la línea base).

- **PAY_2 (Estado de pago del mes 2):** De manera similar, el cliente presenta un valor de **0** en el segundo mes más reciente, indicando nuevamente ausencia de atraso registrado (categoría más frecuente: 52,62%). El modelo asocia este comportamiento con una reducción adicional de **-0,0306** (3,06 puntos porcentuales por debajo de la línea base).

**5. Contexto Global del Modelo**

Los factores que dominan la explicación de este cliente coinciden estrechamente con las variables más importantes a nivel global del modelo. PAY_0, LIMIT_BAL, PAY_AMT1 y PAY_2 se encuentran entre las cinco variables con mayor importancia global media absoluta. Esto indica que el perfil de riesgo de este cliente sigue el patrón típico que el modelo ha aprendido de la población general, sin desviaciones idiosincrásicas significativas.

**6. Nota Metodológica**

Los valores SHAP presentados reflejan asociaciones estadísticas aprendidas por el modelo durante su entrenamiento, no relaciones causales comprobadas. La afirmación de que una variable "empuja" o "reduce" la probabilidad de default debe interpretarse en términos de cómo el modelo pondera dicha variable, no como una relación de causa-efecto en la realidad. Esta explicación debe complementarse con el juicio de expertos humanos y no debe utilizarse como único criterio para la toma de decisiones crediticias.