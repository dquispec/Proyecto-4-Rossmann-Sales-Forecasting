# Proyecto 4: Forecasting de Ventas - Rossmann

Sistema de prediccion de ventas diarias para la cadena alemana Rossmann (drogueria/farmacia). Cuatro niveles que cubren desde EDA descriptivo hasta pipeline MLOps con monitoreo de drift y retraining automatico.

## Contexto

El dataset **Rossmann Store Sales** (Kaggle) contiene ventas diarias de **~1,115 tiendas** de Rossmann durante mas de 2 anos (enero 2013 - julio 2015). El target es Sales.

El dataset incluye 2 archivos principales:
- `train.csv`: ~1 millon de filas (1115 tiendas x ~750 dias) con Sales, Customers, Open, Promo, SchoolHoliday, StateHoliday.
- `store.csv`: atributos por tienda (StoreType, Assortment, CompetitionDistance, Promo2 y campos relacionados).

Particularidad: los dias cerrados registran Sales=0 y deben filtrarse antes del analisis. Las tiendas cierran principalmente los domingos y feriados.

## Estructura del proyecto

```
Proyecto 4 - Rossmann Sales Forecasting/
|-- 01-data/                          # train.csv, store.csv
|-- 02-script/                        # Notebooks Jupyter (.ipynb)
|   |-- 01_basico_eda.ipynb
|   |-- 02_intermedio_ml.ipynb
|   |-- 03_avanzado_boosting.ipynb
|   |-- 04_experto_pipeline.ipynb
|-- 03-resultados/                    # Outputs generados
|-- 04-explicacion del codigo/        # PDFs con explicacion linea por linea
|-- README.md
```

## Niveles del proyecto

| Nivel | Tecnica | Output principal | Resultado clave |
|---|---|---|---|
| Basico | EDA descriptivo (estacionalidad, promociones, segmentos) | 7 graficos + CSV de KPIs | Distribucion log-normal, promociones +32% uplift |
| Intermedio | 5 modelos clasicos (Linear, Ridge, DecisionTree, RandomForest, KNN) + 2 baselines | Comparacion + diagnostico | Random Forest RMSPE 0.13 |
| Avanzado | LightGBM + XGBoost + features de lag (7, 14, 28 dias) + rolling means + SHAP | Mejor modelo + analisis de negocio | LightGBM RMSPE 0.087 |
| Experto | Pipeline MLOps (joblib + encoders + MLflow + API + drift + retraining) | Sistema productizable end-to-end | Drift detectado en Year/Month, retrain v2.0 |

## Hallazgos principales

1. **3 niveles de estacionalidad confirmados**: anual (diciembre pico, enero valle), mensual (variacion ~50%), semanal (lunes y martes mas fuertes, sabado mas bajo).
2. **Promociones generan ~32% de uplift** en ventas promedio. Es la palanca operativa mas importante.
3. **StoreType "b" vende 2-3x mas** que las demas categorias. Diferencia critica que el modelo debe capturar.
4. **Las features de lag dominan el modelo avanzado**: Sales_lag7 y Sales_rmean7 son las features mas importantes. Confirma la regla del forecasting: el pasado reciente es el mejor predictor del futuro inmediato.
5. **El split temporal es obligatorio**: split aleatorio invalida la evaluacion en forecasting.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
# Opcionales:
pip install lightgbm xgboost shap mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) y colocar `train.csv` y `store.csv` en `01-data/`.
2. Abrir cualquiera de los notebooks de `02-script/` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- EDA de series de tiempo y datos transaccionales
- Filtrado y limpieza con foco en datos retail (dias cerrados, outliers)
- Feature engineering temporal: componentes de fecha + lags + rolling means
- Split temporal (critico en forecasting)
- Baselines naive como referencia
- Modelos clasicos (Linear, Ridge, Decision Tree, Random Forest, KNN)
- Boosting (LightGBM, XGBoost) con early stopping
- Hyperparameter tuning ligero por grid
- RMSPE (metrica oficial Kaggle Rossmann)
- Interpretabilidad con feature_importances + SHAP
- Analisis de negocio: uplift de Promo real vs predicho, error por segmento
- MLOps: pipeline con encoders persistidos, API de forecasting, drift PSI + KS, retraining automatico

## Documentacion detallada

Cada notebook tiene un PDF asociado en `04-explicacion del codigo/` con explicacion linea por linea.
