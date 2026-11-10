# Breast Cancer Image Classification Web App

This project preserves the original breast cancer image classification workflow and exposes the saved baseline models through a small research-use web application.

## Scope

- Preserves the original training artifacts and saved checkpoints under `msi_results_full/`
- Uses the original model checkpoint schema and preprocessing settings from the training code
- Exposes only the reliable preserved baselines for inference:
  - Ultrasound
  - Histopathology
- Excludes chest X-ray from the live inference UI because it was not reliable enough for a production-style workflow and was treated as exploratory-only

## App structure

- Backend: [app/main.py](app/main.py)
- Frontend page: [app/templates/index.html](app/templates/index.html)
- Styling: [app/static/styles.css](app/static/styles.css)
- Frontend logic: [app/static/app.js](app/static/app.js)

## Start the app

From the project root:

```powershell
& '.venv/Scripts/python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

## Notes

- This is research-use only and is not a clinical decision support system.
- The app loads the saved baseline checkpoints without retraining or modifying experiment outputs.
- Model metrics are read from the existing saved JSON files in `msi_results_full/`.
