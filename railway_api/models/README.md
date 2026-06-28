# models/

This folder must contain your four XGBoost model files before deploying:

```
models/model_tuned_smote_Boredom.ubj
models/model_tuned_smote_Confusion.ubj
models/model_tuned_smote_Engagement.ubj
models/model_tuned_smote_Frustration.ubj
```

Copy them from the parent Detection/ folder.
If you want to use different model variants, update MODEL_FILES in app.py.
