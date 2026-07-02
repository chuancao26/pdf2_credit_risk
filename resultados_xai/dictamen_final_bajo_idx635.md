# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_bajo_idx635_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una probabilidad de incumplimiento del **24,39%**, significativamente por debajo del umbral de clasificación del **59,00%**. En consecuencia, la clase predicha es **No Default (0)**, lo cual coincide con el resultado real observado (No Default). El modelo acierta en este caso. Como referencia, el modelo presenta un AUC en test de 0,773 y un F1 de 0,533, lo que indica una capacidad discriminativa moderada pero no perfecta.

**2. Referencia Base**

El valor esperado SHAP (línea base) del modelo es **0,3038** (30,38%), que representa la probabilidad promedio de incumplimiento que el modelo asigna a un cliente típico de la población de entrenamiento. La probabilidad de este cliente (24,39%) se sitúa **por debajo** de esa línea base, indicando que, en conjunto, las características del cliente empujan la predicción hacia un riesgo menor que el promedio. La tasa de default real en la población de entrenamiento es del 22,12%, por lo que el cliente se encuentra ligeramente por encima de esa tasa real pero por debajo de la línea base del modelo.

**3. Principales Factores de Riesgo**

Los siguientes factores incrementaron la probabilidad de default predicha por el modelo (efecto positivo en SHAP):

- **PAY_AMT1 (Monto pagado en el mes más reciente)**: El cliente pagó **1.891 NT$** en el mes 1. Este valor se encuentra entre el percentil 25 (1.000 NT$) y la mediana (2.133,5 NT$) de la población, es decir, es un pago relativamente bajo. El modelo asocia este comportamiento con un mayor riesgo, contribuyendo con **+0,0276** a la probabilidad (2,76 puntos porcentuales por encima de la línea base).

- **PAY_AMT6 (Monto pagado en el mes más antiguo)**: El cliente pagó **330 NT$** en el mes 6. Este valor también se sitúa entre el percentil 25 (132,25 NT$) y la mediana (1.500 NT$), siendo un pago bajo. El modelo interpreta este patrón de pagos reducidos en meses anteriores como un factor de riesgo, aportando **+0,021** a la probabilidad (2,10 puntos porcentuales).

**4. Principales Factores Protectores**

Los siguientes factores redujeron la probabilidad de default predicha (efecto negativo en SHAP):

- **PAY_0 (Estado de pago del mes más reciente)**: El cliente presenta un código **-2**, que corresponde a **"pago puntual"** (sin atraso). Esta categoría es poco frecuente en la población (solo el 9,16% de los clientes la presenta). El modelo asocia este excelente historial de pago con una reducción significativa del riesgo, contribuyendo con **-0,0536** a la probabilidad (5,36 puntos porcentuales por debajo de la línea base). Es el factor más influyente en la predicción.

- **PAY_2 (Estado de pago del mes 2)**: El cliente también presenta código **-2** (pago puntual) en este mes. Esta categoría aparece en el 12,56% de la población. El modelo lo considera un fuerte factor protector, con una contribución de **-0,0391** (3,91 puntos porcentuales por debajo de la línea base).

- **BILL_AMT3 (Monto de la factura del mes 3)**: La factura del cliente fue de **1.799 NT$**, un valor muy por debajo del percentil 25 de la población (2.754 NT$). El modelo asocia facturas bajas con menor riesgo, contribuyendo con **-0,0191** (1,91 puntos porcentuales por debajo de la línea base).

**5. Contexto Global del Modelo**

Los principales impulsores de la predicción para este cliente coinciden en gran medida con las variables más importantes a nivel global del modelo. **PAY_0** es la variable con mayor importancia global (SHAP medio absoluto de 0,0701), y también domina la explicación local de este cliente. **PAY_2** y **PAY_AMT1** también figuran entre las cinco variables más importantes globalmente. Sin embargo, **PAY_AMT6** y **BILL_AMT3** tienen una importancia global menor, pero adquieren relevancia en este caso particular debido a los valores específicos del cliente. En conjunto, el perfil de riesgo de este cliente sigue la lógica general del modelo, con un énfasis en el historial de pagos como principal determinante.

**6. Nota Metodológica**

Los valores SHAP presentados reflejan las asociaciones aprendidas por el modelo FT-Transformer durante su entrenamiento, no relaciones causales comprobadas. Un valor SHAP positivo indica que el modelo asocia esa característica con un incremento en la probabilidad de default, mientras que uno negativo indica una asociación con una disminución. Estas atribuciones deben interpretarse como evidencia del razonamiento interno del modelo, no como una evaluación definitiva del cliente. Se recomienda complementar este análisis con juicio experto y considerar el contexto completo del cliente antes de tomar decisiones.