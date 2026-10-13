# cv for fraud detection at verizon

classification and fraud detection for sensitive identity documents (telecom-2a), break through tech ai studio, fall 2026, in partnership with verizon.

## team members

| name | github handle | contribution |
|------|---------------|--------------|
| margaret cartee | @margaret-ca | hog/lbp feature extraction, synthetic fraud generation |
| abdullah suleman | @abdullahsulem | data collection, exploratory data analysis, dataset documentation |
| aarav rajeev gupta | @A-Arav0307 | data preprocessing, feature engineering, data validation |

## project overview

verizon onboards wireless and wireline customers with identity documents such as passports, driver's licenses, and national id cards. some of those documents are forged. the goal of this project is to build a binary classifier that looks at an image of an identity document and labels it genuine or fake, so that identity theft can be caught during onboarding.

planned milestones:

1. binary classification of genuine vs fake documents (current)
2. an ai agent that calls the classifier to flag fraud on submitted documents
3. multi-class classification of document type (passport, driver's license, national id)

current focus: a cnn that distinguishes genuine from fake passports. no model is trained in this repository right now. earlier model-training code was removed so the cnn can be built from scratch.

## data

source: the idnet identity document analysis dataset on kaggle (chitreshkr/idnet-identity-document-analysis), a synthetic dataset of about 597,900 identity document images (about 400 gb) covering passports, driver's licenses from 10 u.s. states, and national id cards from 10 european countries. the full archive is not stored in this repository.

local samples in this repository:

- `idnet_sample/` : full-size sampled images, split into `real/` and `fake/`, then by document type (`passport`, `drivers_license`, `national_id`). passports are from the greek (grc) set, about 100 per class.
- `idnet_sample_compressed/` : the same images re-encoded as webp to save disk space.
- `idnet_processed/` : grayscale 256x256 versions of the compressed images.
- `idnet_processed_manifest.csv` : one row per processed image with its path, label, and document type.
- `data/sample/` : a separate local pull made with `scripts/sample_idnet.py` (not committed). it contains genuine passport images from the `positive` folder of the grc archive.

## what has been done so far

- sampled a small subset of the dataset by document type (`image-slice.py`, `config.py`).
- built a streaming sampler that reads images straight from the kaggle download without fetching the whole archive. it can filter by folder (for example `--include /positive/`) and logs the folders it reaches (`scripts/sample_idnet.py`).
- generated synthetic fake documents from the sampled real ones, with two approaches: whole-image degradation using blur, noise, and compression (`generate-fake.py`), and localized tampering using copy-move and splice patches (`generate-fake-localized.py`).
- compressed the samples to webp (`compress.py`).
- preprocessed images to grayscale and 256x256, and wrote a manifest (`preprocess.py`).
- extracted hog and lbp features (`extract-features.py`).
- explored the data (`exploration.ipynb`) and prototyped the sampling and preprocessing steps in colab (`preprocessing_pipeline.ipynb`, with its model-training cells removed).

## repository layout

- `config.py` : shared paths and constants
- `image-slice.py`, `generate-fake.py`, `generate-fake-localized.py`, `compress.py`, `preprocess.py`, `extract-features.py` : data pipeline, run in that order
- `scripts/sample_idnet.py` : streaming sampler for the kaggle archive
- `exploration.ipynb`, `preprocessing_pipeline.ipynb` : notebooks
- `requirements.txt` : python dependencies

## setup

```
git clone https://github.com/Break-Through-Tech/Telecom-2A-classification-and-fraud-detection-for-sensitive-documents.git
cd Telecom-2A-classification-and-fraud-detection-for-sensitive-documents
pip install -r requirements.txt
```

to pull a sample straight from kaggle, put your kaggle access token at `~/.kaggle/access_token`, install `stream-unzip`, then run for example:

```
python scripts/sample_idnet.py --include /positive/ --max-images 75 --per-folder 75 --max-download-mb 8000
```

the genuine images in the grc archive start about 5 gb into the download.

## next steps

- build and train a cnn for passports that classifies genuine vs fake.
- split train and test by document so that no document and its altered copy appear on both sides.
- keep color and a higher resolution than 256x256, since forged regions can be small.
- evaluate with accuracy, precision, recall, f1, roc-auc, and a confusion matrix.
- later milestones: the ai agent, then multi-class document type classification.
