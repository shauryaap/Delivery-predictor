# Delivery-predictor
# ABC Logistics – Delivery Delay Predictor

Streamlit app that predicts, at dispatch, whether a parcel will miss its delivery window.
Built for the ABC Logistics case (Business Analytics). Model: decision tree trained on 1,000 deliveries.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (single-parcel and batch prediction) |
| `train_model.py` | Trains the decision tree and saves `delay_model.joblib` |
| `delay_model.joblib` | Saved model loaded by the app |
| `delivery_delay.xlsx` | Dataset (1,000 deliveries, 11 features + target) |
| `notebooks/` | Class notebook with all models (logistic, SVM, KNN, decision tree) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Results (test set, 200 parcels)
Decision tree: 100% accuracy, recall 1.00, F1 1.00. Delays in this dataset follow a few simple rules
(warehouse processing time above ~90 min, vehicle age above 7 years), so the tree is small (depth 5, 6 leaves).
