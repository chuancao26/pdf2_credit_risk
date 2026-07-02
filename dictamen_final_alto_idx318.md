# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_alto_idx318_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asignó a este cliente una probabilidad de incumplimiento (default) de **0.5482** (54.82%). El umbral de clasificación óptimo del modelo es **0.5900**. Dado que la probabilidad predicha (0.5482) es inferior al umbral (0.5900), el modelo clasificó al cliente como **no default (clase 0)**. Sin embargo, el resultado real observado fue **default (clase 1)**, por lo que nos encontramos ante un **error de tipo falso negativo**: el modelo subestimó el riesgo real del cliente. Cabe señalar que el margen respecto al umbral es muy estrecho (0.0418 puntos), lo que indica que se trata de un caso limítrofe. El modelo presenta un AUC en test de 0.773 y un F1 de 0.5332, lo que sugiere un rendimiento moderado y que errores como este son esperables dentro de su margen de incertidumbre.

**2. Referencia Base**

El valor base del modelo (*shap_expected_value*) es **0.3038**, que representa la probabilidad promedio de default estimada para la población de entrenamiento. La probabilidad predicha para este cliente (0.5482) se sitúa **0.2444 puntos por encima** de dicha base, lo que indica que el modelo percibe a este cliente como significativamente más riesgoso que el cliente promedio. La tasa de default real en la población de entrenamiento es del 22.12%, mientras que la probabilidad de este cliente duplica con creces ese valor.

**3. Principales Factores de Riesgo (empujan la probabilidad al alza)**

Los cinco factores con mayor impacto en la predicción de este cliente tienen todos **signo positivo**, es decir, incrementaron la probabilidad de default por encima del valor base. Se detallan a continuación ordenados por magnitud de impacto:

- **BILL_AMT1 (Monto de la factura del mes más reciente)**: El cliente presenta un valor de **0 NT$**, muy por debajo del percentil 25 de la población (3,552.5 NT$). El modelo asocia esta ausencia total de facturación con un incremento de **+0.0567** en la probabilidad predicha. Este comportamiento es atípico: tener facturación cero en el mes más reciente es inusual y el modelo lo interpreta como una señal de alerta.

- **PAY_AMT1 (Monto pagado en el mes más reciente)**: El cliente pagó **0 NT$** en el mes 1, valor muy por debajo del percentil 25 poblacional (1,000 NT$). El modelo asocia este nulo pago reciente con un incremento de **+0.0541** en la probabilidad. La ausencia total de pago en el mes más reciente es un fuerte indicador de riesgo según el modelo.

- **PAY_AMT6 (Monto pagado en el mes más antiguo, mes 6)**: El cliente también pagó **0 NT$** en el mes 6, muy por debajo del percentil 25 (132.25 NT$). El modelo asocia este valor con un incremento de **+0.0289** en la probabilidad. La falta de pago incluso en el período más antiguo disponible refuerza la percepción de riesgo.

- **PAY_AMT4 (Monto pagado en el mes 4)**: Nuevamente, el cliente pagó **0 NT$**, valor muy por debajo del percentil 25 (300 NT$). El modelo asocia este valor con un incremento de **+0.0274** en la probabilidad. El patrón de pagos nulos se extiende a múltiples meses.

- **AGE (Edad del cliente)**: El cliente tiene **47 años**, valor que se sitúa por encima del percentil 75 de la población (41 años). El modelo asocia esta edad superior a la media con un incremento de **+0.0240** en la probabilidad. Si bien la edad tiene baja importancia global, para este cliente en particular resultó ser un factor relevante.

**4. Principales Factores Protectores (reducen la probabilidad)**

**No se identifican factores protectores entre los cinco principales.** Todos los factores del *top 5* tienen impacto positivo (incrementan el riesgo). Esto es consistente con un perfil de riesgo elevado, donde el modelo no encuentra elementos compensatorios en las variables más influyentes. Cabe destacar que variables como **PAY_0** (estado de pago del mes 1) y **LIMIT_BAL** (límite de crédito), que globalmente son las más importantes del modelo, presentan un valor SHAP de **0.0** para este cliente, lo que indica que no contribuyeron a la predicción individual.

**5. Contexto Global del Modelo**

El patrón de atribución de este cliente es **atípico** respecto al comportamiento global del modelo. Las dos variables más importantes a nivel global —**PAY_0** (importancia media absoluta: 0.0701) y **LIMIT_BAL** (0.0557)— tienen contribución nula en este caso. En su lugar, los principales impulsores son **BILL_AMT1** y **PAY_AMT1**, que sí figuran entre las variables globalmente relevantes (puestos 4.º y 3.º respectivamente), pero también aparecen **PAY_AMT6**, **PAY_AMT4** y **AGE**, que tienen baja importancia global (puestos 10.º, 16.º y 19.º). Esto sugiere que el modelo está aplicando una lógica particular para este cliente, centrada en la ausencia total de pagos y facturación, en lugar de basarse en su historial de morosidad (PAY_0) o su límite de crédito.

**6. Nota Metodológica**

Los valores SHAP presentados en este informe reflejan **asociaciones aprendidas por el modelo** entre las variables y la probabilidad de default, no relaciones causales demostradas. Que el modelo asocie una edad de 47 años o una facturación de 0 NT$ con un mayor riesgo no implica que dichas características *causen* el incumplimiento. Estas atribuciones deben interpretarse como evidencia del razonamiento interno del modelo, no como diagnóstico financiero. Se recomienda complementar esta evaluación con el juicio de expertos en crédito y considerar el contexto completo del cliente antes de tomar decisiones.