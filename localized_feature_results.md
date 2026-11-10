# Localized Tampering Feature Engineering Results

## Dataset
- Real documents: 397
- Localized fake documents: 397
- Total images: 794
- Document categories: Driver's License, Passport, National ID
- Train/test split: 75% / 25%
- Split method: GroupShuffleSplit
- Matching real and fake documents were kept in the same split group.

## Feature Extraction
- HOG: 8,100 features per image
- LBP: 10 histogram features per image
- Image size: 256 x 256 grayscale

## Model Results

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| HOG + Logistic Regression | 87.50% | 94.08% |
| LBP + Logistic Regression | 65.50% | 63.05% |
| HOG + Decision Tree | 90.50% | 90.50% |

## Findings
- HOG + Decision Tree achieved the highest accuracy (90.50%).
- HOG + Logistic Regression achieved the highest ROC-AUC (94.08%).
- LBP + Logistic Regression performed weakest, with 65.50% accuracy.
- The previous 100% results did not hold on this harder localized-tampering dataset.

## Notes
These are preliminary results from one grouped 75/25 train/test split.
Further validation, including grouped cross-validation and per-tampering-type
evaluation, is needed before treating these as final model performance.
