# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_bajo_idx635_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una **probabilidad de incumplimiento del 24,39%**, significativamente por debajo del umbral de clasificación del **59,00%**. En consecuencia, la **clase predicha es 0 (no default)**, lo cual coincide con el resultado real observado (clase real = 0). El margen de separación respecto al umbral es amplio (34,61 puntos porcentuales), lo que indica una predicción robusta. Como referencia de fiabilidad global, el modelo alcanza un AUC de 0,773 y un F1 de 0,533 en el conjunto de test.

**2. Referencia Base**

El valor esperado SHAP (*baseline*) del modelo es **0,3038 (30,38%)**, que representa la probabilidad promedio de default sobre la población de entrenamiento. La probabilidad de este cliente (24,39%) se sitúa **5,99 puntos porcentuales por debajo** de dicho baseline. En comparación con la tasa base de default poblacional (22,12%), el cliente presenta un riesgo ligeramente superior al promedio de la población, pero el modelo lo considera de bajo riesgo debido al efecto neto de sus factores protectores.

**3. Principales Factores de Riesgo (SHAP positivo — incrementan la probabilidad)**

- **PAY_AMT1 (Monto pagado en el mes más reciente, mes 1):** El cliente pagó **1.891 NT$** en el mes más reciente. Este valor se sitúa entre el percentil 25 (1.000 NT$) y la mediana (2.133,5 NT$) de la población. El modelo asocia este monto de pago moderado con un **incremento de +0,0276** en la probabilidad de default respecto al baseline. Aunque no es un pago bajo en términos absolutos, el modelo pondera este nivel como ligeramente indicativo de riesgo.

- **PAY_AMT6 (Monto pagado en el mes más antiguo, mes 6):** El cliente pagó **330 NT$** en el mes más antiguo. Este valor se encuentra entre el percentil 25 (132,25 NT$) y la mediana (1.500 NT$), siendo un pago relativamente bajo. El modelo asocia este comportamiento con un **incremento de +0,0210** en la probabilidad. Cabe destacar que esta variable es un factor de riesgo atípico para este cliente, ya que globalmente tiene una importancia baja en el modelo (puesto 17 de 23).

**4. Principales Factores Protectores (SHAP negativo — disminuyen la probabilidad)**

- **PAY_0 (Estado de pago del mes más reciente, mes 1):** El cliente presenta el código **-2** en esta variable, una categoría poco frecuente (9,16% de la población). Este código, que no corresponde a las categorías estándar de atraso, es interpretado por el modelo como un fuerte indicador de bajo riesgo, reduciendo la probabilidad en **-0,0536**. Es el factor más influyente de toda la explicación.

- **PAY_2 (Estado de pago del mes 2):** El cliente también presenta código **-2** en esta variable (12,56% de la población). El modelo asocia esta categoría con una **reducción de -0,0391** en la probabilidad de default.

- **BILL_AMT3 (Monto de la factura del mes 3):** El cliente tiene una factura de **1.799 NT$** en el mes 3, un valor **por debajo del percentil 25** de la población (2.754 NT$). El modelo interpreta este bajo saldo como un factor protector, reduciendo la probabilidad en **-0,0191**.

**5. Contexto Global del Modelo**

Tres de los cinco factores principales de este cliente (PAY_0, PAY_2 y PAY_AMT1) se encuentran entre las cinco variables más importantes del modelo a nivel global, lo que indica que el perfil de riesgo de este cliente sigue el patrón típico que el modelo ha aprendido. Sin embargo, la variable PAY_AMT6 aparece como un factor relevante solo para este cliente, siendo de importancia marginal en el conjunto global. Esto sugiere que, aunque el modelo utiliza su lógica general, existen particularidades locales en la atribución para este caso concreto.

**6. Nota Metodológica**

Los valores SHAP presentados reflejan **asociaciones estadísticas aprendidas por el modelo** entre las variables del cliente y la probabilidad de default predicha. No deben interpretarse como relaciones causales demostradas. Esta explicación describe el razonamiento interno del modelo, no la realidad del cliente. Las decisiones crediticias deben complementarse con juicio experto y no basarse exclusivamente en puntuaciones automatizadas.