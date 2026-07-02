# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_medio_idx4095_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una **probabilidad de incumplimiento (default) del 38.53%**. Dado que el umbral de clasificación óptimo se sitúa en **59.00%**, la predicción del modelo es **clase 0 (no default)**. El resultado real del cliente fue efectivamente **no default**, por lo que la predicción es correcta. El margen respecto al umbral es de 20.47 puntos porcentuales, lo que indica una decisión moderadamente clara, no limítrofe. El modelo presenta un AUC de 0.773 y un F1 de 0.5332 en el conjunto de prueba, lo que debe considerarse al interpretar su fiabilidad general.

**2. Referencia Base**

El valor base del modelo (*shap expected value*) es **30.38%**. Esto representa la probabilidad promedio de default que el modelo asigna a un cliente promedio de la población de entrenamiento, antes de considerar las características individuales. La probabilidad de este cliente (38.53%) se sitúa **8.15 puntos porcentuales por encima** de dicha base, y también por encima de la tasa base de default poblacional (22.12%). Esto indica que el modelo percibe a este cliente como de riesgo superior al promedio, aunque no lo suficiente para clasificarlo como default.

**3. Principales Factores de Riesgo**

A continuación se detallan las variables que, según el modelo, **incrementaron** la probabilidad de default respecto al valor base:

- **LIMIT_BAL (Monto de crédito otorgado):** El cliente cuenta con un límite de crédito de **20,000 NT$**, un valor **muy por debajo del percentil 25** de la población (50,000 NT$). El modelo asocia un límite de crédito tan bajo con un mayor riesgo, contribuyendo con **+0.1659** a la probabilidad. Este es, con diferencia, el factor de riesgo más influyente para este cliente.

- **PAY_AMT1 (Monto pagado en el mes más reciente):** El cliente pagó **1,183 NT$** en el mes 1. Este valor se sitúa entre el percentil 25 (1,000 NT$) y la mediana (2,133.5 NT$), lo que el modelo interpreta como un pago relativamente bajo, contribuyendo con **+0.0465** al riesgo.

- **PAY_AMT3 (Monto pagado en el mes 3):** El pago de **1,200 NT$** en el mes 3 también se encuentra entre el percentil 25 (400 NT$) y la mediana (1,879 NT$). El modelo asocia este nivel de pago con un riesgo ligeramente mayor, aportando **+0.0179**.

**4. Principales Factores Protectores**

Las siguientes variables **disminuyeron** la probabilidad de default según el modelo:

- **PAY_0 (Estado de pago del mes más reciente):** El cliente presenta un estado de pago de **0**, que corresponde a "pago mínimo o uso de crédito revolvente sin atraso registrado". Esta es la categoría más común en la población (49.21%). El modelo asocia esta ausencia de morosidad con un menor riesgo, reduciendo la probabilidad en **-0.0618**. Es el factor protector más relevante.

- **MARRIAGE (Estado civil):** El cliente es **soltero**, la categoría más frecuente en la población (53.58%). El modelo asocia este estado civil con un perfil de menor riesgo, contribuyendo con **-0.0184** a la baja de la probabilidad.

**5. Contexto Global del Modelo**

Los principales impulsores de la predicción para este cliente coinciden en gran medida con las variables más importantes a nivel global. **LIMIT_BAL**, **PAY_0** y **PAY_AMT1** son, respectivamente, el segundo, primero y tercer factor más influyentes del modelo en toda la población. Esto indica que el perfil de riesgo de este cliente sigue la lógica general del modelo. No obstante, se observa una ligera atipicidad: las variables **MARRIAGE** y **PAY_AMT3** tienen un peso local mayor para este cliente del que tienen en el promedio global, lo que sugiere que el modelo está utilizando estas características de forma más intensa para explicar el riesgo de este caso concreto.

**6. Nota Metodológica**

Los valores SHAP presentados en este análisis reflejan las **asociaciones internas aprendidas por el modelo** entre las variables y la probabilidad de default. No deben interpretarse como relaciones causales demostradas en la realidad. Esta explicación describe cómo el modelo pondera cada característica para llegar a su predicción, y no constituye una evaluación definitiva del cliente. Las decisiones basadas en este modelo deben complementarse con el juicio de expertos humanos y con información adicional no contemplada en el mismo.