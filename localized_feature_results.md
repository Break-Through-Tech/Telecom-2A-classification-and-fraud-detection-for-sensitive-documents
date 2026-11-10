# Localized Tampering Feature Engineering Results

## Dataset

- Real documents: 397
- Fake documents: 397
- Total images: 794
- Categories: drivers_license, national_id, passport
- Feature extraction: HOG (8,100 features) and LBP (10 features)
- Evaluation: 5-fold Stratified Group Cross-Validation
- Original and fake versions of each document remained in the same fold.

## Tampering Types

| Tampering Type | Number of Fake Images |
|---|---:|
| Copy-move | 195 |
| Splice | 202 |

## Overall Model Performance

| Model                     |   Mean Accuracy |   Accuracy Std |   Mean ROC-AUC |   ROC-AUC Std |
|:--------------------------|----------------:|---------------:|---------------:|--------------:|
| HOG + Logistic Regression |          0.8891 |         0.0145 |         0.9390 |        0.0110 |
| LBP + Logistic Regression |          0.6386 |         0.0230 |         0.6443 |        0.0240 |
| HOG + Decision Tree       |          0.9005 |         0.0146 |         0.9005 |        0.0146 |

## Performance by Tampering Type

| Tampering Type   | Model                     |   Mean Accuracy |   Accuracy Std |   Mean ROC-AUC |   ROC-AUC Std |
|:-----------------|:--------------------------|----------------:|---------------:|---------------:|--------------:|
| copy_move        | HOG + Logistic Regression |          0.8926 |         0.0244 |         0.9426 |        0.0162 |
| copy_move        | LBP + Logistic Regression |          0.6736 |         0.0288 |         0.6325 |        0.0395 |
| copy_move        | HOG + Decision Tree       |          0.8953 |         0.0224 |         0.8606 |        0.0416 |
| splice           | HOG + Logistic Regression |          0.9006 |         0.0208 |         0.9322 |        0.0248 |
| splice           | LBP + Logistic Regression |          0.6638 |         0.0251 |         0.6127 |        0.0319 |
| splice           | HOG + Decision Tree       |          0.9131 |         0.0175 |         0.8805 |        0.0233 |

## Key Findings

1. HOG + Logistic Regression achieved the highest overall ROC-AUC (0.9390).
2. HOG + Decision Tree achieved the highest overall accuracy (0.9005).
3. HOG features substantially outperformed LBP features.
4. HOG + Logistic Regression achieved ROC-AUC values of 0.9426 for copy-move and 0.9322 for splicing.
5. Copy-move appears slightly more challenging for the HOG models, although the differences are modest.

## Limitations

- Results use one generated dataset and five grouped cross-validation folds.
- Results are preliminary and should not be treated as performance on unseen real-world fraud.
- The copy-move and splice evaluations use different fake subsets and should be interpreted separately.
- Further testing on independent documents and additional tampering techniques is needed.
