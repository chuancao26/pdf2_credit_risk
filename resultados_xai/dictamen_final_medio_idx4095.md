# DICTAMEN TÉCNICO DE EXPLICABILIDAD
**Archivo Origen:** cliente_riesgo_medio_idx4095_xai.json
**Intentos de Certificación:** 1/3
---

**1. Resumen de la Predicción**

El modelo FT-Transformer asigna a este cliente una probabilidad de incumplimiento (default) del **38.53%**. Dado que el umbral de clasificación óptimo se sitúa en **59.00%**, la predicción del modelo es **clase 0 (no default)**. Esta predicción **coincide con el resultado real observado** (el cliente efectivamente no incurrió en default). El margen respecto al umbral es de 20.47 puntos porcentuales, lo que indica una decisión moderadamente clara, no limítrofe. Cabe señalar que el modelo presenta un AUC de 0.773 y un F1 de 0.5332 en el conjunto de prueba, lo que proporciona un contexto de fiabilidad general.

**2. Referencia Base**

El valor base del modelo (*shap expected value*) es de **30.38%**. Este valor representa la probabilidad promedio de default que el modelo asigna a un cliente "promedio" de la población sobre la que fue entrenado, antes de considerar las características individuales. La probabilidad predicha para este cliente (38.53%) se sitúa **8.15 puntos porcentuales por encima de dicha base**, lo que indica que el modelo percibe un riesgo superior al promedio poblacional. Para contextualizar, la tasa de default real en el conjunto de entrenamiento fue del **22.12%**.

**3. Principales Factores de Riesgo**

Los siguientes factores incrementaron la probabilidad de default predicha por el modelo, desplazándola por encima del valor base:

- **LIMIT_BAL (Monto de crédito otorgado):** El cliente dispone de un límite de crédito de **20,000 NT$**, un valor **muy por debajo del percentil 25** de la población (50,000 NT$). El modelo asocia este límite extremadamente bajo con un mayor riesgo, contribuyendo con **+0.1659** a la probabilidad. Este es, con diferencia, el factor de riesgo más influyente para este cliente.

- **PAY_AMT1 (Monto pagado en el mes más reciente):** El cliente pagó **1,183 NT$** en el mes 1. Este valor se sitúa **entre el percentil 25 y la mediana** de la población (1,000 y 2,133.5 NT$, respectivamente). El modelo interpreta este pago relativamente bajo como un factor de riesgo, aportando **+0.0465** a la probabilidad.

- **PAY_AMT3 (Monto pagado hace tres meses):** El cliente pagó **1,200 NT$** en el mes 3. Similar al caso anterior, este valor se encuentra **entre el percentil 25 y la mediana** (400 y 1,879 NT$, respectivamente). El modelo asocia este comportamiento de pago moderado con un riesgo adicional, contribuyendo con **+0.0179**.

**4. Principales Factores Protectores**

Los siguientes factores redujeron la probabilidad de default predicha, contrarrestando parcialmente los factores de riesgo:

- **PAY_0 (Estado de pago del mes más reciente):** El cliente presenta un valor de **0** en esta variable, lo que indica **pago mínimo o uso de crédito revolvente sin atraso registrado**. Esta es la categoría más frecuente en la población de entrenamiento (49.21%). El modelo asocia este historial de pago sin demoras con un menor riesgo, restando **-0.0618** a la probabilidad.

- **MARRIAGE (Estado civil):** El cliente es **soltero**. Esta es la categoría más común en la población (53.58%). El modelo asocia ligeramente este estado civil con un perfil de menor riesgo, contribuyendo con **-0.0184** a la reducción de la probabilidad.

**5. Contexto Global del Modelo**

Tres de los cinco factores principales que explican la predicción de este cliente —LIMIT_BAL, PAY_0 y PAY_AMT1— coinciden con las variables más importantes a nivel global del modelo (primer, segundo y tercer lugar en importancia media absoluta SHAP). Esto indica que el perfil de riesgo de este cliente sigue un patrón **típico y esperable** según la lógica general del modelo. Las variables MARRIAGE y PAY_AMT3, aunque aparecen en el top-5 local, tienen una importancia global menor, pero sus contribuciones son pequeñas en magnitud.

**6. Nota Metodológica**

Los valores SHAP presentados en este análisis son atribuciones **correlacionales** que reflejan cómo el modelo pondera internamente cada característica para llegar a su predicción. No deben interpretarse como relaciones causales comprobadas entre las variables y el comportamiento real de pago del cliente. Esta explicación describe el razonamiento del modelo, no la realidad del cliente. Las decisiones crediticias deben complementarse con el juicio de expertos humanos y no basarse únicamente en puntuaciones automatizadas.