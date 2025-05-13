# ML Job Postings NLP Project

## 📄 Project Overview
This project applies Natural Language Processing (NLP) techniques to analyze 1,000 machine learning job postings across the United States. The analysis includes topic modeling, sentiment analysis, named entity recognition, trend detection, and interactive dashboarding.

## 🎯 Objectives
- Extract and analyze dominant topics in job descriptions.
- Understand sentiment trends and tone in job postings.
- Extract skills and named entities using NER.
- Visualize job trends over time, regions, companies, and seniority.
- Build an interactive exploratory dashboard inside the notebook.

## 🗂 Dataset
- **Source**: Proprietary dataset of 1,000 ML job postings.
- **Columns**:
  - `job_posted_date`
  - `company_address_locality`
  - `company_address_region`
  - `company_name`
  - `company_website`
  - `company_description`
  - `job_description_text`
  - `seniority_level`
  - `job_title`

## 🔧 Tools & Libraries
- Python 3
- Pandas, Numpy
- Scikit-learn (LDA, Clustering)
- TextBlob (Sentiment Analysis)
- spaCy (NER)
- UMAP-learn, TSNE (Clustering visualization)
- Matplotlib, Seaborn, Plotly
- WordCloud
- Ipywidgets (Dashboard)

## Project Layout
```bash
ML_job_posting_NLP/
│
├── data/                         # 📁 Raw and processed datasets
│   ├── 1000_ml_jobs_us.csv
│   └── sample_ml_job_postings.csv
│
├── notebooks/                    # 📁 Jupyter/Google colab notebooks
│   ├── NLP_ML_Job_Postings.ipynb
│
├── dashboard/                    # 📁 Dashboard code (ipywidgets and Streamlit)
│   ├── dashboard_widgets.py
│   └── dashboard_streamlit.py 
│
├── reports/                      # 📁 Reports, insights, presentation material
│   └── insights_summary.pdf 
│
├── visuals/                      # 📁 Saved charts, plots, wordclouds
│   └── topic_wordcloud.png 
│
├── models/                       # 📁 Trained models, embeddings (to be added later)
│
├── requirements.txt              # 📄 Python package dependencies
│
└── README.md                     # 📄 Project documentation

```


## 📊 Key Analyses
- Exploratory Data Analysis (EDA)
- Topic Modeling using LDA
- Sentiment Analysis using TextBlob
- Named Entity Recognition (NER) with spaCy
- Clustering job descriptions using KMeans + UMAP/TSNE
- Trend Analysis of keywords, topics, sentiment over time
- Interactive Dashboard (Ipywidgets + Plotly)

## 📈 Business Insights
- Majority of ML jobs are posted in California and New York.
- Python, Machine Learning, Data Science are dominant keywords.
- Overall sentiment of postings is highly positive.
- Clustering reveals distinct job types such as Research, Engineering, Data Science.

## ⚠ Limitations
- Dataset is limited to 1,000 postings.
- No salary or benefits data.
- BERTopic not used due to environment restrictions (optional future work).

## 🚀 Future Enhancements
- Use BERTopic for dynamic topic modeling over time.
- Extract skill trends using NLP-powered skill extractors.
- Deploy interactive dashboard using Streamlit, Voila, or Panel.
- Predict emerging skill demand using forecasting models.

## 📚 References
- HuggingFace Datasets
- Scikit-learn Documentation
- TextBlob Documentation
- SpaCy NER Models
- UMAP-learn for dimensionality reduction

## 🤝 Acknowledgements
Thanks to all open-source contributors and dataset providers.
