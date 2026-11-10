# Application and tests

FastAPI backend is `app/main.py`; the UI is under `app/templates` and `app/static`. The application supports ultrasound and histopathology only, loads preserved checkpoints in `msi_results_full`, validates image dimensions/format and modality via a detector, rejects unsupported/mismatched modality, then predicts using the selected classifier. Errors include unavailable checkpoints/detector and invalid input. Chest X-ray is marked unavailable. Do not interpret confidence as calibrated certainty absent calibration evidence.

`tests/test_backend_modality_guard.py` is a unit/backend guard test; `smoke_test_outputs/` indicates a smoke-test artifact directory. Outcomes were not run for this task and no clinical validation is documented. Unit tests and smoke tests are software checks, not clinical validation.
