# Proyecto 4: Forecasting de Ventas - Rossmann (Nivel Experto)

Pipeline MLOps end-to-end para forecasting de ventas diarias de la cadena Rossmann. Incluye pipeline con encoders persistidos, API de forecasting, tracking con MLflow y monitoreo de drift con retraining automatico.

## Contexto

El dataset **Rossmann Store Sales** (Kaggle) contiene ventas diarias de **~1,115 tiendas** de Rossmann durante mas de 2 anos (enero 2013 - julio 2015). El target es Sales.

El dataset incluye 2 archivos principales:
- `train.csv`: ~1 millon de filas (1115 tiendas x ~750 dias) con Sales, Customers, Open, Promo, SchoolHoliday, StateHoliday.
- `store.csv`: atributos por tienda (StoreType, Assortment, CompetitionDistance, Promo2 y campos relacionados).

Particularidad: los dias cerrados registran Sales=0 y deben filtrarse antes del analisis. Las tiendas cierran principalmente los domingos y feriados.

## Contenido del repositorio

```
Proyecto-4-Rossmann-Sales-Forecasting/
|-- 04_experto_pipeline.ipynb    # Pipeline MLOps completo
|-- README.md
```

## Pipeline experto

| Componente | Descripcion | Resultado |
|------------|-------------|-----------|
| Pipeline atomico | Encoders + LightGBM persistidos con joblib | Sin data leakage, versionable |
| Features temporales | Lags (7, 14, 28 dias) + rolling means | Sales_lag7 = feature mas importante |
| Tracking | MLflow (registro de metricas y parametros) | Experimentos reproducibles |
| API de forecasting | Funcion que imita endpoint de prediccion | Scoring por tienda y fecha |
| Drift monitoring | PSI + KS sobre Year/Month | Drift detectado, retrain v2.0 |
| Retraining automatico | Trigger por umbral de drift PSI | Sistema autosostenible |

## Hallazgos principales

1. **3 niveles de estacionalidad confirmados**: anual (diciembre pico, enero valle), mensual (variacion ~50%), semanal (lunes y martes mas fuertes, sabado mas bajo).
2. **Promociones generan ~32% de uplift** en ventas promedio. Es la palanca operativa mas importante.
3. **StoreType "b" vende 2-3x mas** que las demas categorias. Diferencia critica que el modelo debe capturar.
4. **Las features de lag dominan**: Sales_lag7 y Sales_rmean7 son las mas importantes. El pasado reciente es el mejor predictor del futuro inmediato.
5. **El split temporal es obligatorio**: split aleatorio invalida la evaluacion en forecasting.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy lightgbm xgboost shap mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) y colocar `train.csv` y `store.csv` en la misma carpeta del notebook.
2. Abrir `04_experto_pipeline.ipynb` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- Feature engineering temporal: lags + rolling means + componentes de fecha
- Split temporal correcto para forecasting
- Pipeline con encoders persistidos (joblib)
- Tracking de experimentos con MLflow
- API de forecasting simulada
- Monitoreo de drift PSI + Kolmogorov-Smirnov
- Retraining automatico con versionado
- Metrica RMSPE (oficial Kaggle Rossmann)
