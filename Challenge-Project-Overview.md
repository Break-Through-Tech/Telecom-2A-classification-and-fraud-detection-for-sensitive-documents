
# Classification to detect fraud of sensitive identity documents

**Company / Org:** Verizon Inc 
**Challenge Advisor:** DJ (Dhananjaya Ramachandra), Dhananjaya.Ramachandra@verizon.com  
**AI Coach:** Srihari Kamath, srihari.kamath@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Verizon
Verizon is a global leader in telecommunications, providing wireless, fiber-optic, and digital services to millions of customers.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use the open source data sets pertaining to "identity documents (example: passport, driving id etc)" to build models leveraging ML/DNN algorithms for binary classification and accordingly detect fraud! Thus, these models would help in eliminating identity theft as part of wireless/wireline customer onboarding and help save millions of dollars in lost revenue for the company!

### Success Criteria
Successful completion of the aforementioned project milestones leveraging ML/DNN algorithms and large language models.

### Stretch Goals
- AI agent which can do autonomous fraud detection i.e. detect if the submitted document of the user is a genuine document or a fake document by invoking the binary classifier that was built as part of the core/non-stretch goal.

- The AI agent could also optionally support various analytics and ad-hoc queries on the identity document corpus.

### Extra Stretch Goals
- Detect the type of document that was submitted i.e. classify (thru a simple multi-class classifier or a cluster-er) if the document submitted is a passport or driving id or national id or some unknown document!

NOTE: This is a extra-stretch goal for the students to experience the multi-class classifier apart from the binary classifier they built as part of the core/non-stretch goal!

### Project Milestones
Use these milestones to guide your work and accordingly create GitHub Project board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September 2026 | Binary Classification | • Business and problem framing<br>• Data collection and preparation<br>• Feature engineering<br>• Model engineering and selection<br>• Model evaluation and reports |
| October 2026 | Autonomous AI Agent | • AI agent creation for autonomous classification and accordingly detect fraud|
| November 2026 | Multi-Class Classification | • Data preparation<br>• Model engineering and selection<br>• Model evaluation and reports |


> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** IDNet Identity Document Analysis Dataset (Kaggle)
**Format:** Static Images (`.jpg`, `.png`) categorized by folder structure
**Size:** ~400 GB total (Students sample down to < 1 GB / ~600 images locally)
**Location:** https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis


### Key Details

- **Description of Data**
  - Contains **597,900 synthetically generated images** (~400 GB total) of identity documents designed for privacy-preserving document analysis and classification.
  - Covers three primary document categories across multiple regions: **Passports**, **Driver's Licenses** (10 U.S. states), and **National ID Cards** (10 European countries).

- **Known Limitations & Preprocessing Needed**
  - **Initial Storage & RAM Slicing:** The full dataset is ~400 GB; students can use Python's [`glob` module](https://docs.python.org/3/library/glob.html) to slice local file arrays (e.g., `file_list[:200]`) to cap RAM under 1 GB and speed up training [see scikit-learn's `train_test_split` guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
  - **Synthetic Fake Generation:** Since the dataset contains only genuine document templates, students can generate the "Fake" subset using OpenCV Gaussian Blur [`cv2.GaussianBlur`](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) with Gaussian noise injection, or use industry-standard libraries like [Albumentations Documentation](https://albumentations.ai/docs/).
  - **Feature Extraction & Normalization:** Images must be grayscaled via OpenCV [`cv2.cvtColor`](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html) and resized prior to computing spatial edge histograms or micro-texture patterns using [`scikit-image feature module`](https://scikit-image.org/docs/stable/api/skimage.feature.html) (specifically `skimage.feature.hog` and `skimage.feature.local_binary_pattern`).

- **Documentation & Links**
  - **Kaggle Dataset Page:** [IDNet: Identity Document Analysis Image Dataset](https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis)
  
---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Computer Vision, Deep Learning / Neural Networks, Natural Language Processing (NLP), Large Language Models (LLMs) / Generative AI

```markdown
```text
[1. Load Directory Data] 
           │
           ▼
[2. Feature Extraction]  ──> (Grayscale ──> Resize ──> Compute HOG/LBP Vectors)
           │
           ▼
[3. Data Partitioning]   ──> (Train/Test Split 80/20)
           │
           ▼
[4. Model Training]      ──> (Classification algorithms / Fit)
           │
           ▼
[5. Inference]           ──> (Predict on Unseen Documents ──> Class Probability)
           │
           ▼
[6. Quality Evaluation]  ──> (Precision, Recall, F1, Confusion Matrix)
```

**Recommended Libraries:**
- numpy, cv2, midv500, skimage, sklearn, os, glob etc

**Evaluation Metrics:**
- accuracy, precision, recall, F1-score, and ROC-AUC

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- **[IDNet Benchmark Dataset Paper (arXiv)](https://arxiv.org/abs/2408.01690):** The official research paper introducing the IDNet dataset, detailing synthetic identity document generation and privacy-preserving document fraud analysis.
- **[PyImageSearch - Document & Image Classification Overview](https://pyimagesearch.com/):** Comprehensive articles covering classical computer vision techniques and real-world document processing workflows.

**Technical Tutorials:**
- **[Histogram of Oriented Gradients (HOG) & SVM Tutorial](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/):** Step-by-step guide on calculating HOG descriptors and training Linear/RBF SVM classifiers for object recognition.
- **[Local Binary Patterns (LBP) Texture Classification](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/):** Hands-on tutorial on extracting LBP micro-texture histograms with OpenCV and training Random Forest classifiers.

**Code Examples:**
- **[Robust Document OCR & Preprocessing Pipeline (Kaggle)](https://www.kaggle.com/code/ahmedmohamedab/robust-document-ocr-preprocessing-pipeline):** Kaggle notebook demonstrating document binarization, perspective warping, and image cleanup pipelines.
- **[scikit-image Feature Extraction Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html):** Official documentation and working code snippets for `skimage.feature.hog` and `skimage.feature.local_binary_pattern`.

**Other:**
- **[OpenCV Image Processing Tutorials](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html):** Official guide on core image transformations, Gaussian smoothing, and adaptive thresholding methods.
- **[MIDV-500 Dataset Overview (YouTube)](https://www.youtube.com/):** Search for "MIDV-500 dataset" on YouTube for visual demonstrations of document capture under extreme real-world lighting and angle distortions.

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Email; please copy your teammates and AI Studio Coach
* If something cant be answered via email, then we could schedule a zoom/google meet call.
* Note: DJ will be OOO starting Aug 24th to Sep 11 2026. So, please reach out to your AI Studio Coach with urgent questions.

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
