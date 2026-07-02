# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_alto_idx318_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asignó a este cliente una probabilidad de incumplimiento (default) de **0.5482** (54.82%). El umbral de clasificación óptimo (F1-maximizado) es **0.5900**, por lo que el modelo clasificó al cliente como **no default (clase 0)**. Sin embargo, el resultado real observado fue **default (clase 1)**, lo que constituye un **falso negativo** (error tipo II). El margen respecto al umbral es muy estrecho (0.0418), indicando que la predicción fue limítrofe. Las métricas globales del modelo en test son AUC = 0.773 y F1 = 0.5332, lo que refleja una capacidad discriminativa moderada pero con una tasa de error no despreciable.

**2. Referencia Base**

El valor esperado SHAP (línea base) del modelo es **0.3038**, que representa la probabilidad promedio de default que el modelo asigna a un cliente típico de la población de entrenamiento. La tasa base de default real en el conjunto de entrenamiento es **22.12%**. La probabilidad predicha para este cliente (0.5482) se sitúa muy por encima de ambos valores de referencia, lo que indica que el modelo percibe un riesgo significativamente elevado, aunque no lo suficiente para cruzar el umbral de clasificación.

**3. Principales Factores de Riesgo (SHAP positivo — empujan la probabilidad al alza)**

Los cinco factores con mayor impacto en la predicción tienen todos signo positivo, es decir, incrementaron la probabilidad de default respecto a la línea base. Se detallan a continuación ordenados por magnitud de impacto:

- **BILL_AMT1 (Monto de la factura del mes más reciente)**: El cliente presenta un valor de **0 NT$**, lo que se encuentra muy por debajo del percentil 25 de la población (3,552.5 NT$) y es extremadamente bajo respecto a la media poblacional (51,715.23 NT$). El modelo asocia esta ausencia de facturación con un incremento de **+0.0567** en la probabilidad de default.

- **PAY_AMT1 (Monto pagado en el mes más reciente)**: El cliente pagó **0 NT$**, valor muy por debajo del percentil 25 poblacional (1,000 NT$) y de la media (5,682.81 NT$). El modelo interpreta este nulo pago como un factor de riesgo que incrementa la probabilidad en **+0.0541**.

- **PAY_AMT6 (Monto pagado en el mes 6, el más antiguo)**: Nuevamente **0 NT$**, muy por debajo del percentil 25 (132.25 NT$) y de la media (5,325.8 NT$). Contribuye con **+0.0289** al alza de la probabilidad.

- **PAY_AMT4 (Monto pagado en el mes 4)**: El cliente presenta **0 NT$**, valor muy inferior al percentil 25 (300 NT$) y a la media (4,908.35 NT$). Aporta **+0.0274** al incremento de riesgo.

- **AGE (Edad del cliente)**: El cliente tiene **47 años**, valor que supera el percentil 75 poblacional (41 años) y está muy por encima de la mediana (34 años). El modelo asocia una edad relativamente avanzada con un aumento de **+0.0240** en la probabilidad de default.

Es importante destacar que el cliente presenta un historial de pagos impecable (todos los PAY_* con valor -2, que indica pago puntual), y sin embargo el modelo pondera como factores de riesgo los montos nulos de facturación y pago en múltiples meses, junto con una edad superior a la media poblacional.

**4. Principales Factores Protectores (SHAP negativo — reducen la probabilidad)**

**No se identifican factores protectores entre los cinco principales impulsores de la predicción.** Todos los factores del top-5 tienen impacto positivo, lo que significa que el modelo no encontró ninguna característica en este cliente que actuara como mitigante significativo del riesgo dentro de ese conjunto. Este patrón es consistente con un perfil de riesgo elevado donde todos los factores relevantes apuntan en la misma dirección.

**5. Contexto Global del Modelo**

El patrón de atribución de este cliente resulta **atípico** respecto al comportamiento global del modelo. A nivel global, la característica más importante es **PAY_0** (estado de pago del mes más reciente), con una importancia media absoluta SHAP de 0.0701. Sin embargo, para este cliente, PAY_0 tiene un valor SHAP de **0.0**, lo que indica que no contribuyó en absoluto a la predicción. En contraste, características con baja importancia global como **PAY_AMT6** (rango global #10), **PAY_AMT4** (rango #16) y **AGE** (rango #19) se convierten en los principales impulsores locales. Esto sugiere que el modelo está utilizando una lógica diferente para este cliente en particular, basada en la combinación de pagos nulos y edad avanzada, en lugar de los patrones de morosidad que suelen dominar globalmente.

**6. Nota Metodológica**

Los valores SHAP presentados en este informe reflejan **asociaciones estadísticas aprendidas por el modelo** entre las características del cliente y la probabilidad de default, no relaciones causales comprobadas. Que el modelo asocie un valor bajo de facturación o pago con mayor riesgo no implica que dichas variables causen el incumplimiento. Esta explicación debe complementarse con juicio experto humano y no debe utilizarse como único criterio para la toma de decisiones crediticias. El modelo tiene una precisión limitada (AUC=0.773) y, como se observa en este caso, puede cometer errores de clasificación, especialmente en clientes con perfiles limítrofes respecto al umbral de decisión.