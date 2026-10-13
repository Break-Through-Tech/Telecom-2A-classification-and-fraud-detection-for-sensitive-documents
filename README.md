# CV for Fraud Detection at Verizon

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Margaret Cartee    | @margaret-ca | HOG/LBP feature extraction, synthetic fraud generation, model training            |
| Abdullah Suleman   | @abdullahsulem    | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Aarav Rajeev Gupta     | @A-Arav0307 | Data preprocessing, feature engineering, data validation                 |

---

## 🎯 **Project Highlights**

**Example:**

- Built an end-to-end binary classification pipeline -- sampling, synthetic fraud generation, preprocessing, HOG/LBP feature extraction, and a Random Forest classifier -- to detect tampered identity documents (passports, driver's licenses, national ID cards) for Verizon's customer-onboarding fraud-prevention use case.
- Compared two synthetic fraud-simulation strategies using stratified 5-fold cross-validation: whole-image degradation (blur/noise/compression) produced a perfect but misleading 1.000 ROC-AUC, while realistic localized tampering (copy-move and splice forgery) dropped to a near-chance ~0.60 ROC-AUC.
- Delivered engineering finding for the fraud-detection task: global, whole-image texture/edge descriptors (HOG/LBP) are not sufficient to catch realistic, localized document forgery -- a block-based self-similarity approach is needed next (see Next Steps).

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

1. **Clone the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Access the dataset** ([IDNet on Kaggle](https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis)) -- either:
   - Configure the Kaggle API (`pip install kaggle`, place `kaggle.json` per Kaggle's instructions) and let `image-slice.py` download it automatically, or
   - Manually download the archive and place it at the path set by `ARCHIVE_PATH` in `config.py` (works regardless of file extension).

   Adjust `CATEGORIES` and `STATE_FILTER` in `config.py` to control which document types/country-state codes get sampled.

4. **Run the pipeline in order** from the project root:
   ```bash
   python image-slice.py              # stream-sample real document images from the archive
   python generate-fake.py            # OR generate-fake-localized.py -- synthesize the "fake" class
   python compress.py                 # compress samples to WebP, replacing the raw copies
   python preprocess.py               # grayscale + resize to a fixed size, write the manifest CSV
   python extract-features.py         # compute HOG + LBP feature vectors
   python train-classifier.py         # train/evaluate the Random Forest classifier
   ```

---

## 🏗️ **Project Overview**

**Describe:**

This project was completed as part of the **Break Through Tech AI Studio** program (Fall 2026), in partnership with **Verizon**. Our Challenge Advisor was DJ Ramachandra (Verizon), with AI Studio Coach Srihari Kamath.

**Objective:** build a binary classifier that distinguishes genuine identity documents (passports, driver's licenses, national ID cards) from fraudulent/tampered ones, to support identity verification during wireless/wireline customer onboarding.

**Real-world significance:** identity document fraud during customer onboarding enables identity theft and costs telecom providers like Verizon significant lost revenue. An automated first-pass screening classifier could flag suspicious document submissions for human review, reducing fraud losses and protecting customers from identity theft.

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

- **Feature extraction**: for each preprocessed grayscale image, computed a **Histogram of Oriented Gradients (HOG)** descriptor (9 orientations, 16×16 pixel cells, 2×2 cell blocks) and a **Local Binary Pattern (LBP)** texture histogram (radius 1, 8 points, uniform method, normalized), concatenated into one ~8,110-dimension feature vector per image.
- **Classifier:** a `RandomForestClassifier` (200 trees, scikit-learn) -- a strong baseline for high-dimensional tabular descriptor features that doesn't require feature scaling.
- **Training/evaluation setup:** an 80/20 stratified train/test split, plus stratified **5-fold cross-validation** across the full dataset (both seeded for reproducibility), since the feature count (~8,110) is roughly 10x the sample count (~794 images) -- exactly the kind of high-dimensional, low-sample regime where a single split can't be trusted on its own.
- **Two fraud-simulation strategies were built and compared as separate experiments:**
  1. `generate-fake.py` -- randomized **whole-image degradation** (Gaussian/motion blur, Gaussian noise, brightness/contrast shifts, JPEG re-compression, dropout patches) via Albumentations.
  2. `generate-fake-localized.py` -- realistic **localized forgery**: either copy-move (cloning a patch from elsewhere in the same image) or splice (copying a patch from a different document), covering only ~8-20% of the image while leaving the rest pixel-identical to the source.


---

## 📈 **Results & Key Findings**

| Fake-generation strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV accuracy (mean ± std) |
|---|---|---|---|---|---|---|
| Whole-image degradation (blur/noise/compression) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 ± 0.003 |
| Localized tampering (copy-move / splice) | 0.491 | 0.485 | 0.405 | 0.441 | 0.594 | 0.519 ± 0.019 (ROC-AUC 0.614 ± 0.034) |

**Key finding 1 -- the "perfect" score was not meaningful.** Cross-validation confirmed the 1.000 score for whole-image degradation was stable across every fold, ruling out a lucky train/test split. But it doesn't indicate real fraud-detection ability: the classifier learned to detect if the image has been globally degraded in some way, a signal that blur, noise, and compression all reliably leave across broad HOG/LBP statistics.

**Key finding 2 -- realistic, localized tampering exposes a blind spot.** Switching to small, localized edits (copy-move and splice, only ~1-3.5% of pixels changed per image) collapsed performance to near chance. This makes sense structurally: HOG/LBP pool statistics over the *entire* image into one vector, so a small localized change is diluted to be nearly invisibile. Copy-move is a particularly hard case -- the "moved" content was already present in the image, so the global feature histogram barely changes at all; no whole-image descriptor can resolve "this patch looks suspiciously identical to that other patch" without directly comparing regions *within* an image to each other.

**Feature importance supports both findings:** in the degradation experiment, importance was concentrated (and later spread, once degradation severity was randomized); in the localized-tampering experiment, the top feature's importance dropped to 0.0036 across ~8,110 features -- essentially no discriminative signal anywhere.

---

## 🚀 **Next Steps**

- **Broaden document coverage.** Add more country/state licenses and passports given the passport-category bug found this milestone.
- **With more time/resources:** explore a CNN-based tamper-*localization* model rather than a single whole-image classification, since document forgery is fundamentally a localization problem.
- **Validate beyond IDNet.** Test against an established forensic benchmark (e.g. CASIA, the Columbia splicing dataset) or real tampered-document examples, to check whether these findings generalize beyond IDNet's synthetic templates.

---

## 📝 **License**

This project is suggested to be licensed under the [MIT License](https://opensource.org/licenses/MIT), pending final approval from our Challenge Advisor.

---

## 📄 **References** (Optional but encouraged)

- [scikit-image `feature` module documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html) (`hog`, `local_binary_pattern`).
- [Albumentations documentation](https://albumentations.ai/docs/) -- used for both the whole-image degradation and tamper-generation pipelines.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank you to our Challenge Advisor, DJ Ramachandra (Verizon), for the project guidance, and to our AI Studio Coach, Srihari Kamath, for ongoing support throughout the program. Thank you to the Break Through Tech AI Studio program for the opportunity.
