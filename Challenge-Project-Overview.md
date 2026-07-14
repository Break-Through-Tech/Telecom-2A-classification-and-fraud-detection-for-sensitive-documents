---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The project is centered on Python, as it leverages ML/DNN algorithms and is intended to run in a Pythonic environment (e.g., Google Colab). |
| Data Readiness | 🟡 | The data requires cleaning and preprocessing; students may spend significant time on these tasks to ensure data quality. |
| Resource Check | 🟢 | Utilizes free-tier tools like Google Colab, which are accessible to students. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The project concept is strong. However, consider refining the data preprocessing steps to ensure students are not overwhelmed. The balance between classification and fraud detection needs to be clearly delineated in the deliverables.

---

# Classification and fraud detection for sensitive documents

**Company / Org:** Verizon  
**Challenge Advisor:** Dhananjaya Ramachandra, Dhananjaya.Ramachandra@verizon.com  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Verizon
Verizon is a global leader in telecommunications, providing wireless, fiber-optic, and digital services to millions of customers. The team aims to leverage advanced AI technologies to enhance security measures and protect the company against financial losses associated with identity theft.

---

## 🎯 The Challenge
### Project Summary
This project focuses on building robust classification and fraud detection models for sensitive identity documents, such as passports and driver's licenses. By utilizing open-source document datasets, the team will implement ML/DNN algorithms and LLMs to secure customer onboarding processes, ultimately preventing identity fraud and safeguarding corporate revenue.

### Success Criteria
Successful completion of the project milestones (classification, fraud detection, and AI agent creation) leveraging ML/DNN algorithms and large language models.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Ingest the MIDV-500 dataset, perform exploratory data analysis, and establish cleaning protocols for image and categorical data. |
| **October** | Feature Engineering & Baseline Modeling | Develop classification features and train initial baseline models to identify document types and detect anomalies. |
| **November** | Model Optimization & Evaluation | Execute iterative hyperparameter tuning, run model verification, and assess the accuracy of fraud detection architectures. |
| **December** | Insights, Deliverables & Presentation | Integrate findings into a final AI agent, generate business recommendations, and finalize the project documentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** MIDV-500 Dataset (https://smartengines.com/wp-content/uploads/2020/04/datasets-of-id-documents-midv-500.pdf)  
**Format:** Images (.png, .jpg) and categorical metadata  
**Size:** under 1gb  
**Location:** Accessible via the provided link; students must mirror a local copy for processing.  

### Key Details
- Publicly available identity documents (passport, driving ID, etc.) including categorical and image data (.png, .jpg) from https://smartengines.com/wp-content/uploads/2020/04/datasets-of-id-documents-midv-500.pdf. Data requires cleaning and preprocessing.
- Preprocessing must include image normalization, noise reduction, and the mapping of categorical labels to numerical vectors for training input.

---

## 🛠️ Suggested Approach
**ML Problem Type:** Classification, NLP & RAG, Multi-Agent Systems  
**Recommended Libraries:**
- ML/DNN algorithms
- Large Language Models (LLMs)
- Generative AI
- Natural Language Processing (NLP)
- Computer Vision
- Deep Learning / Neural Networks
- Google Colab
**Evaluation Metrics:** Precision, Recall, F1-Score for classification, and Area Under the ROC Curve (AUC) for fraud detection sensitivity.

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- Research on Document Identity Verification and Fraud Detection patterns in telecommunications.
**Technical Tutorials:**
- PyTorch and TensorFlow tutorials for Image Classification and Computer Vision basics.
**Code Examples:**
- Sample Jupyter notebooks for image preprocessing and standard CNN architectures.

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and scheduled lab sessions  
**Response time:** 48-72 hours  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
