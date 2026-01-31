![Diagrama](docs/images/miro_tmp.png)

![Diagrama](docs/images/seeFFT.png)


# 1. Resumen del Proyecto

Mientras que el análisis técnico tradicional se basa en indicadores fijos y la mayoría de los modelos de Deep Learning sufren de "miopía temporal", este proyecto propone una solución de **Aprendizaje por Refuerzo Profundo (DRL)** diseñada para capturar patrones cíclicos multiescala en el mercado Forex (EURUSD). El desafío central es la extracción de características en series temporales financieras, donde el ruido suele sepultar a la señal.

## El núcleo de la solución:

El pipeline se sustenta en tres innovaciones arquitectónicas integradas para maximizar la robustez del agente:

### Extracción Temporal Multiescala (TimesNet Backbone):
En lugar de procesar el precio como una secuencia lineal simple, el modelo utiliza un backbone de **TimesNet**. Mediante la **Transformada Rápida de Fourier (FFT)**, el sistema identifica los periodos dominantes en el histórico y remodela los datos 1D en tensores 2D. Esto permite que kernels de convolución (Inception blocks) capturen variaciones intra-periodo e inter-periodo simultáneamente, detectando estacionalidades que los modelos RNN o Transformers estándar suelen omitir.

### Aprendizaje Híbrido "Self-Rewarding" (SRDDQN):
Para acelerar la convergencia en el espacio de recompensas dispersas del trading, el agente utiliza una arquitectura de doble cabezal:
- **Q-Head:** Estima el valor de las acciones (Short, Neutral, Long).
- **Reward-Head:** Una rama auto-supervisada que predice el potencial de ganancia futuro.
El sistema implementa una lógica **Self-Rewarding**, donde la recompensa de entrenamiento es un híbrido entre la "intuición" del modelo y una etiqueta experta, mitigando el sesgo de la red y forzando al agente a aprender no solo qué hacer, sino por qué es valioso hacerlo.

### Integración de Alphas y Contexto Dual:
El modelo no solo observa el precio horario. Su estado se compone de un contexto dual (24 horas recientes + 24 días previos) y se enriquece con una capa de **indicadores Alpha pre-validados** (procedentes de la fase anterior de minería de estrategias). Esto proporciona al agente una ventaja informativa (Edge), combinando el aprendizaje de representaciones crudas con conocimiento experto estadístico.

## Evaluación

El éxito del agente no se mide por el retorno total, sino por la **calidad del retorno**. El sistema de validación prioriza el **Sortino Ratio** y el **Calmar Ratio** mediante un esquema de **Rolling Validation**, asegurando que el modelo no sea simplemente un buscador de suerte, sino un gestor de riesgo eficiente capaz de batir al *Buy & Hold* en entornos de alta volatilidad.

# 2. Requisitos, Ejecución y Artefactos

## Datos necesarios (inputs)

Para ejecutar el proyecto se requieren:
- `EURUSD_Final_Dataset_RL_With_All_Alphas.csv`: Dataset procesado que incluye precios OHLC (Ask/Bid), normalizaciones Z-score y las columnas de indicadores Alpha calculadas mediante Numba.
- Ficheros de configuración de hiperparámetros (Config dict).

## Artefactos generados (outputs)

- **Checkpoints del Modelo:** Archivos `.pth` con los pesos de la red `SRDDQN_Net`.
- **Logs de Entrenamiento:** Registro detallado de pérdida híbrida, evolución de Epsilon y métricas de validación por episodio.
- **Curvas de Equity:** Comparativas visuales entre el rendimiento del modelo, el *Buy & Hold* y el coste acumulado de swaps y comisiones.

# 3. Metodología y Arquitectura

El proyecto se divide en capas de abstracción para garantizar la fidelidad financiera:

## Lógica de Trading (TradingLogic Class)
Un simulador de broker ECN/STP estricto que gestiona:
- **Costes de Ejecución:** Spread real, comisiones por operación y **Swap triple** los miércoles.
- **Contabilidad Financiera:** Diferenciación entre *Balance* y *Equity* (Mark-to-Market) para evitar la toma de decisiones basada en PnL no realizado.

## Etiquetado Experto (Expert Labeling)
Se implementó un oráculo con "visión de futuro" (look-ahead) limitado a $m$ periodos para generar etiquetas de recompensa ideales. Estas etiquetas sirven como guía supervisada para que el agente aprenda a identificar los puntos de máximo potencial (Min-Max future potential) antes de intentar operar en entornos totalmente inciertos.

## Arquitectura de Red (Two-Headed Monster)
- **Hourly Encoder:** TimesNet enfocado en la micro-estructura y el momentum.
- **Daily Encoder:** TimesNet enfocado en el régimen de mercado y niveles de soporte/resistencia diarios.
- **Fusion Layer:** Un MLP que integra los embeddings temporales con el estado escalar del agente (posición actual, margen, fees e indicadores Alpha).

# 4. Conclusiones y Futuro

Este framework demuestra que el Deep Reinforcement Learning, cuando se combina con arquitecturas de visión aplicadas a series temporales (TimesNet), puede encontrar ineficiencias en mercados altamente eficientes. El uso de aprendizaje híbrido reduce drásticamente el tiempo de entrenamiento y mejora la estabilidad de la política del agente.

## Próximos pasos:

1.  **Expansión Multi-Activo y Universos de Alphas:** Evolucionar de un solo par a una arquitectura multivariante que integre correlaciones entre divisas (e.g., USD Index) y una librería de Alphas más extensa, similar a la desarrollada en proyectos previos.
2.  **Transición a Vision Transformers (ViT):** Sustituir el backbone de CNN por **Vision Transformers** sobre las representaciones 2D generadas por FFT. Esto permitiría capturar dependencias globales de largo alcance mediante mecanismos de auto-atención, superando las limitaciones de los campos receptivos de las convoluciones.
3.  **Regime-Based Curriculum Learning:** Implementar un esquema de entrenamiento progresivo donde el agente comience operando en regímenes de baja volatilidad y tendencias claras ("fáciles") antes de ser expuesto a crisis financieras o mercados laterales ruidosos ("difíciles"), acelerando la formación de una política base robusta.
4.  **Optimización Bayesiana de Hiperparámetros:** Realizar un ajuste fino sistemático de la arquitectura (d_model, top_k) y parámetros de RL (gamma, tau, learning rate) mediante **Optuna**, buscando maximizar el Sortino Ratio y minimizar el Maximum Drawdown en el set de validación.
5.  **Adaptación Incremental vía Meta-Learning (DoubleAdapt):** Integrar un framework de meta-aprendizaje inspirado en **DoubleAdapt** para mitigar el *concept drift*. Esto permitiría al agente adaptar dinámicamente sus representaciones de datos (*feature adapter*) y sus expectativas de recompensa (*label adapter*) ante cambios bruscos en la distribución de los retornos del mercado, permitiendo un aprendizaje continuo sin olvidar patrones previos. (https://arxiv.org/pdf/2306.09862)
