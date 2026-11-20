# Smart Waste Segregation

A local Flask + SQLite waste sorting app with MobileNetV2 transfer learning. The supplied Garbage Classification ZIP was extracted to `dataset/garbage_classification`; its original folder names are kept as the model classes.

## Dataset included

| Original class folder | Images |
| --- | ---: |
| battery | 945 |
| biological | 985 |
| brown-glass | 607 |
| cardboard | 891 |
| clothes | 5,325 |
| green-glass | 629 |
| metal | 769 |
| paper | 1,050 |
| plastic | 865 |
| shoes | 1,977 |
| trash | 697 |
| white-glass | 775 |
| **Total** | **15,515** |

The original ZIP is left untouched in the user's Downloads folder. `model/class_names.json` records the sorted class names used by the trainer and Flask predictor.

## Install dependencies

Use Python 3.10 or 3.11, then from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is required for training and predictions. If installation fails on this machine, use the error's recommended TensorFlow/Python version pairing; the Flask app still starts without a trained model and gives a friendly message on image analysis.

## Train

The script searches under `dataset/` for the class folders, preserves their original names, uses 224×224 images, a seeded 80/20 train/validation split, light training augmentation, MobileNetV2 ImageNet weights and batch size 32. It caches the frozen MobileNetV2 image features once, then fits the small classifier head; this keeps CPU-only training practical. EarlyStopping and ModelCheckpoint keep the best validation-loss model. If a temporary local ZIP copy exists, the default command uses it to avoid OneDrive per-file sync overhead; the extracted project dataset is kept intact.

```powershell
python train_model.py
```

Optional custom extraction directory:

```powershell
python train_model.py --dataset-dir "path\to\extracted\dataset"
```

Outputs are `model/waste_model.keras` and `model/class_names.json`. Training accuracy and validation accuracy are printed from evaluation of the final best-weight model; results are not prefilled or simulated. Training can take a while on a laptop, and no accuracy is promised before it completes.

## Run Flask

```powershell
python app.py
```

Open http://127.0.0.1:5000. SQLite creates `database/waste.db` and tables automatically. Flask loads `model/waste_model.keras` once when the process starts. Restart Flask after training to load the new model. If confidence is under 50%, the result page marks the prediction as uncertain.

### Development admin account

The first app startup creates a local development admin if it does not already exist:

- Email: `admin@example.com`
- Password: `admin123`

New registrations always receive the `user` role. Set `FLASK_SECRET_KEY` to a private value outside a local demo.

## Files

- `app.py` — Flask routes, auth, SQLite, one-time model loading and inference
- `train_model.py` — dataset discovery, augmentation, MobileNetV2 training and evaluation
- `dataset/garbage_classification/` — extracted image dataset (ignored by Git because it is large)
- `model/class_names.json` — original class mapping
- `model/waste_model.keras` — created after training
- `templates/`, `static/` — responsive HTML, CSS and JavaScript

Chart.js and web fonts use public CDNs; model inference runs locally and no paid API is used.
