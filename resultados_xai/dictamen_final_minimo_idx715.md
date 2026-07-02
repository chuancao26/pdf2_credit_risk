# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_minimo_idx715_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una **probabilidad de incumplimiento (default) del 2,69%** (0,0269). El umbral de clasificación óptimo (maximizando F1 en validación) es del 59,00% (0,5900). Dado que 2,69% << 59,00%, el modelo predice **clase 0 (no default)** con un margen muy amplio de 56,31 puntos porcentuales. La clasificación es correcta: el cliente efectivamente no incurrió en default (clase real = 0). El modelo presenta un AUC de 0,773 y un F1 de 0,5332 en el conjunto de test, lo que indica una capacidad discriminativa moderada.

**2. Referencia Base**

El valor esperado SHAP (shap_expected_value) es **0,3038**, que representa la probabilidad promedio de default que el modelo asigna a un cliente promedio de la población de entrenamiento (es decir, el punto de partida antes de aplicar los ajustes individuales de cada variable). La tasa base de default en el conjunto de entrenamiento es del 22,12%. La probabilidad de este cliente (2,69%) se sitúa muy por debajo tanto del valor base del modelo (30,38%) como de la tasa poblacional (22,12%), lo que refleja un perfil de riesgo significativamente menor que el promedio.

**3. Principales Factores de Riesgo (SHAP positivo — empujan la probabilidad al alza)**

- **BILL_AMT3 (Monto de la factura del mes 3):** El cliente presenta un saldo de factura de **209.680 NT$** en el mes 3 (hace tres meses). Este valor se ubica muy por encima del percentil 75 poblacional (61.346,75 NT$), en el cuartil superior. El modelo asocia este alto monto de factura con un incremento en el riesgo: el valor SHAP de **+0,0369** indica que esta variable empuja la probabilidad de default 3,69 puntos porcentuales por encima del valor base. Es el único factor de riesgo entre los cinco principales.

**4. Principales Factores Protectores (SHAP negativo — empujan la probabilidad a la baja)**

Cuatro de los cinco factores principales actúan como protectores, reduciendo sustancialmente la probabilidad de default:

- **PAY_AMT1 (Monto pagado en el mes 1):** El cliente pagó **100.000 NT$** en el mes más reciente, un valor extremadamente alto (20 veces el percentil 75 de 5.006,25 NT$). El modelo asocia este pago elevado con una fuerte reducción del riesgo: valor SHAP de **-0,0989**, que reduce la probabilidad en 9,89 puntos porcentuales respecto al valor base.

- **LIMIT_BAL (Límite de crédito):** El cliente dispone de un límite de **360.000 NT$**, muy por encima del percentil 75 (240.000 NT$). El modelo interpreta un límite alto como señal de solvencia: valor SHAP de **-0,0969**, reduciendo la probabilidad en 9,69 puntos porcentuales.

- **PAY_0 (Estado de pago del mes 1):** El cliente presenta un estado de pago **0** en el mes más reciente, que corresponde a "pago mínimo/uso de crédito revolvente sin atraso registrado". Esta es la categoría más frecuente en la población (49,21%). El modelo asocia este comportamiento de pago sin retraso con una reducción del riesgo: valor SHAP de **-0,0755**, reduciendo la probabilidad en 7,55 puntos porcentuales.

- **PAY_2 (Estado de pago del mes 2):** Similar al anterior, el cliente presenta estado **0** en el mes 2, también la categoría más común (52,62%). El modelo lo asocia con una reducción adicional del riesgo: valor SHAP de **-0,0306**, reduciendo la probabilidad en 3,06 puntos porcentuales.

**5. Contexto Global del Modelo**

Los factores que dominan la explicación de este cliente coinciden en gran medida con las variables más importantes a nivel global del modelo. PAY_0, LIMIT_BAL y PAY_2 se encuentran entre las cinco variables con mayor importancia global (media de |SHAP|). PAY_AMT1 y BILL_AMT3 tienen una relevancia ligeramente superior en este cliente que en el promedio de la población, pero el patrón general es consistente con la lógica típica del modelo: historial de pagos puntuales, límite de crédito elevado y pagos recientes altos son los principales predictores de bajo riesgo.

**6. Nota Metodológica**

Los valores SHAP presentados reflejan asociaciones aprendidas por el modelo durante su entrenamiento, no relaciones causales comprobadas. Que el modelo asocie un pago elevado con menor riesgo no implica que aumentar el pago cause una reducción del riesgo real. Estas atribuciones deben interpretarse como evidencia de cómo el modelo pondera cada variable para llegar a su predicción, y deben complementarse con juicio experto y análisis adicional antes de tomar decisiones crediticias. Ningún cliente debe ser reducido únicamente a su puntuación de riesgo.