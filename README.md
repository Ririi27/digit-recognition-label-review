# Handwritten Digit Recognition + Label Quality Review

A Python project with two goals: (1) classify handwritten digits, and (2) use the model to **find mislabeled data**, the same kind of quality check used when reviewing annotations for AI training.

**Dataset:** scikit-learn `digits` – 1,797 grayscale images (8x8), 10 classes, roughly balanced (174-183 per class). Bundled with scikit-learn.

## Workflow
1. **EDA** – sample images per digit, class balance.
2. **Model comparison** – KNN, Logistic Regression, Random Forest and SVM with stratified 5-fold CV.
3. **Test evaluation** – held-out 20% test set, classification report, confusion matrix.
4. **Label review** – 5% of labels (89 images) were deliberately corrupted. Out-of-fold predictions (each image scored by a model that never trained on it) rank images by how much the model distrusts the given label. The top-ranked images are flagged for human review.
5. **Agreement** – Cohen's kappa between two models and the true labels, similar to inter-annotator agreement.

## Results
| Model | 5-fold CV accuracy |
|---|---|
| SVM (RBF) | 98.1% |
| KNN | 97.6% |
| Random Forest | 97.5% |
| Logistic Regression | 96.8% |

- **SVM test accuracy: 97.5%.** Most common confusions: 8 predicted as 1 and 9 predicted as 7.
- **Label review:** flagging the 89 most suspicious images caught **83 of the 89 corrupted labels (93.3% precision and recall)**. Flagging the top 178 caught all 89 (100% recall, 50% precision).
- **Cohen's kappa:** SVM vs labels 0.982, Random Forest vs labels 0.977, SVM vs Random Forest 0.974.

## Run it
```bash
pip install -r requirements.txt
python digit_label_review.py
```
Outputs: charts (`*.png`), `model_comparison.csv`, `results.json`, `run_output.txt`.

## Limitations
The original dataset labels are clean, so the label-noise results come from **artificially injected errors**. The images are small (8x8), so this is a learning project, not a production system. Real annotation errors can be subtler than random wrong labels.

**Tech:** Python, NumPy, Pandas, Scikit-learn, Matplotlib
