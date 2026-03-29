# Medical Diagnosis via Semantic Similarity — Deep Learning Benchmark
**Python | BERT | Sentence Transformers | GloVe | Word2Vec | Healthcare AI**

A comparative deep learning research project evaluating four text embedding architectures for predicting patient diagnoses from natural language symptom descriptions. The best-performing model achieved **85% accuracy** using Sentence Transformers (all-MiniLM-L6-v2) with a Support Vector Machine classifier.

Built as a capstone research project (Distinction — 76) in the Deep Learning unit of the Master of Software Engineering (AI) at Torrens University, following the CRISP-DM methodology.

---

## Research Hypothesis

> Utilising semantic similarity for symptom descriptions to determine disease diagnosis can be accurate and effective enough to assist clinicians in triaging patients.

---

## Models Compared

Four embedding architectures were evaluated, each combined with multiple classifiers:

| Model | Best Accuracy | Best Classifier |
|---|---|---|
| **Sentence Transformer (all-MiniLM-L6-v2)** | **85%** | SVM (rbf / linear kernel) |
| Word2Vec | 78% | Logistic Regression |
| BERT | 67% | Gradient Boosting |
| GloVe | 62% | Logistic Regression |

**Winner: Sentence Transformer + SVM** outperformed all other architectures on this clinical dataset.

---

## Full Results Table

| Classifier | all-MiniLM-L6-v2 | GloVe | Word2Vec | BERT |
|---|---|---|---|---|
| Gradient Boosting | 0.56 | 0.32 | 0.49 | 0.67 |
| SVM (poly/rbf/sigmoid/linear) | **0.83/0.85/0.84/0.85** | 0.59 | 0.70 | 0.61 |
| KNeighbours / Random Forest | 0.78 | 0.36 | 0.70 | 0.50 |
| Logistic Regression | 0.83 | 0.62 | 0.78 | 0.67 |

---

## Dataset

The dataset was developed by Figure Eight and is publicly available via:
- [Kaggle — Medical Speech Transcription and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)
- [HuggingFace — medical_asr_recording_dataset](https://huggingface.co/datasets/Hani89/medical_asr_recording_dataset)

The dataset comprises thousands of natural language symptom descriptions contributed by human annotators (e.g. "I need help with my migraines" for the symptom "headache"), covering prevalent medical symptoms with corresponding disease labels.

> Note: At time of publication, no prior research had utilised this dataset, making this one of the first benchmarking studies on it.

---

## Methodology

The CRISP-DM (Cross-Industry Standard Process for Data Mining) framework was followed across six phases:

1. **Business Understanding** — Problem definition: can AI assist clinicians in triaging patients from symptom descriptions?
2. **Data Understanding** — Dataset exploration using WordCloud and statistical analysis via Google Colab
3. **Data Preparation** — Text cleaning, tokenisation, encoding, and label preparation
4. **Modelling** — Four embedding models × four classifier types = comprehensive benchmark
5. **Evaluation** — Accuracy comparison across all model/classifier combinations
6. **Deployment** — Feedback loop implementation; public survey on model trust and accuracy

---

## Notebooks

| Notebook | Embedding Model |
|---|---|
| `Symptoms_Disease_SentenceTransformer.ipynb` | all-MiniLM-L6-v2 (best performing) |
| `Symptoms_Disease_BERT.ipynb` | BERT |
| `Symptoms_Disease_Word2Vec.ipynb` | Word2Vec |
| `Symptoms_Disease_GloVe.ipynb` | GloVe |

---

## Ethical Considerations

Patient privacy, data security, and regulatory compliance were addressed throughout the research, including:
- Use of anonymised, publicly available data
- Transparency in model limitations
- Bias mitigation in training data
- Clinician oversight as a required safeguard

---

## Public Survey Results

A user survey was conducted to assess trust and perceived accuracy of the model:

- **Average rating: 4.33 / 5**
- **76% of respondents** willing to share anonymous data to improve AI accuracy
- Key concerns: accuracy, patient privacy, trustworthiness
- Key assurances cited: clinician oversight, improvements in AI technology

Selected feedback:
- *"Seems accurate enough that I would use it and trust it."*
- *"It made an accurate prediction based on my symptoms."*

---

## Requirements

```bash
pip install sentence-transformers scikit-learn pandas numpy torch transformers gensim
```

Run notebooks in Google Colab or a local Jupyter environment.

---

## Key Findings

- Sentence Transformers significantly outperformed traditional word embeddings (GloVe, Word2Vec) for clinical symptom matching
- SVM with linear and rbf kernels consistently outperformed tree-based classifiers on this task
- BERT underperformed relative to the lightweight Sentence Transformer, suggesting that larger models are not always better for domain-specific clinical NLP tasks
- Public trust in AI-assisted diagnosis is cautiously positive when clinician oversight is maintained

---

## Author

**Kim Trieu**
Applied AI Engineer | LLM Evaluation | Healthcare AI
[LinkedIn](https://www.linkedin.com/in/ktrieu/)

*Part of a broader research focus on AI evaluation and validation in clinical healthcare environments.*
