# Pneumonia Detection from Chest X-Rays

A neural network (CNN / MobileNetV2 transfer learning) that classifies chest X-rays as
**NORMAL** or **PNEUMONIA**, with Grad-CAM heatmaps showing which regions drove the prediction.
The dataset is fetched **directly from Kaggle by the code** - no manual download.


## Files
- `main.py` - everything: Kaggle download, training, evaluation, prediction + Grad-CAM
- `requirements.txt` - dependencies
- `README.md` - this file

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(~1.2 GB, downloaded once via `kagglehub` and cached). If Kaggle asks for credentials, create an API
token (Kaggle > Settings > API) and set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Usage
```bash
python main.py train    --model mobilenet     # or --model cnn (baseline)
python main.py evaluate --model mobilenet
python main.py predict  some_xray.jpeg --model mobilenet
```
Train options: `--epochs`, `--fine-tune-epochs`, `--no-fine-tune`.
Evaluate/predict option: `--threshold` (lower than 0.5 = fewer missed pneumonia cases, more false alarms).

Outputs: model in `models/`, curves / confusion matrix / ROC / metrics / Grad-CAM images in `results/`.

## Google Colab / Jupyter
Set the runtime to GPU, then paste the contents of `main.py` into one cell and run it. In the next cells:
```python
main(["train", "--model", "mobilenet"])
main(["evaluate", "--model", "mobilenet"])
main(["predict", "path/to/xray.jpeg", "--model", "mobilenet"])
```
(Install first with `!pip install -q kagglehub scikit-learn`.)

## How it works
1. Images resized to 224x224; 80/20 train/validation split of `train/`; `test/` used only for final evaluation.
2. Class weights handle the ~3:1 pneumonia:normal imbalance; mild augmentation (small rotation/zoom/shift/contrast).
3. MobileNetV2 (ImageNet weights): train new head, then fine-tune top layers at a low learning rate.
4. Early stopping and checkpointing on validation AUC.
5. Grad-CAM on the last convolutional layer for explainability.

## Results
Evaluated on the dataset's official test set (624 images: 234 NORMAL, 390 PNEUMONIA) with the
MobileNetV2 model, decision threshold 0.5:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 0.97 | 0.61 | 0.75 | 234 |
| PNEUMONIA | 0.81 | 0.99 | 0.89 | 390 |

| Overall | Value |
|---|---|
| Accuracy | 0.85 |
| Macro avg F1 | 0.82 |
| Weighted avg F1 | 0.84 |
| ROC AUC | 0.9643 |

**Interpretation**
- The model catches almost every pneumonia case (recall 0.99), which is the priority in a screening setting.
- The trade-off is false alarms: only 61% of healthy X-rays are recognised as normal, so roughly 4 in 10 normal
  images are flagged as pneumonia (precision on PNEUMONIA is 0.81).
- ROC AUC of 0.96 shows the model separates the classes well; the weaker NORMAL recall comes largely from the
  0.5 threshold and the known distribution shift between this dataset's train and test sets. Raising the
  threshold (e.g. `--threshold 0.7`) trades some pneumonia recall for fewer false alarms.

Plots (confusion matrix, ROC curve, training curves) are saved in `results/`.

## Limitations
- Pediatric images from one hospital; may not generalise to adults or other scanners.
- Small, distribution-shifted test set (624 images) - test accuracy can be lower than validation.
- Check Grad-CAM outputs rather than trusting accuracy alone.

Training on CPU is slow - use a GPU, Google Colab or Kaggle Notebooks.
