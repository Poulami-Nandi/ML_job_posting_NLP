import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Use the following command to run it
# streamlit run dashboard_streamlit.py

# Load data (ensure it's cleaned)
df = pd.read_csv('data/1000_ml_jobs_us.csv')

# Preprocessing
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'], errors='coerce')
df['month'] = df['job_posted_date'].dt.to_period('M')

# Sidebar
st.sidebar.title("ML Job Postings Dashboard")
view = st.sidebar.selectbox("Choose View", [
    'Sentiment Distribution',
    'Monthly Posting Trend',
    'Word Cloud',
    'Region Distribution',
    'Company Sentiment Distribution'
])

# Views
if view == 'Sentiment Distribution':
    st.title("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts(normalize=True)
    st.bar_chart(sentiment_counts)

elif view == 'Monthly Posting Trend':
    st.title("Monthly Job Posting Trend")
    trend = df.groupby('month').size()
    st.line_chart(trend)

elif view == 'Word Cloud':
    st.title("Word Cloud of Job Descriptions")
    all_words = ' '.join(df['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

elif view == 'Region Distribution':
    st.title("Top Regions by ML Job Postings")
    top_regions = df['company_address_region'].value_counts().head(10)
    st.bar_chart(top_regions)

elif view == 'Company Sentiment Distribution':
    st.title("Company-wise Sentiment Distribution")
    top_companies = df['company_name'].value_counts().head(5).index
    company_sentiment = df[df['company_name'].isin(top_companies)].groupby(['company_name', 'sentiment']).size().unstack().fillna(0)
    st.bar_chart(company_sentiment)
