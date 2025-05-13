import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_sentiment_distribution():
    df['sentiment'].value_counts(normalize=True).plot(kind='bar', color='purple', figsize=(8,4))
    plt.title('Sentiment Distribution in Job Descriptions')
    plt.show()

def plot_top_topics():
    for topic in top_topics:
        print(topic)

def plot_sentiment_trend():
    df['job_posted_date'] = pd.to_datetime(df['job_posted_date'], errors='coerce')
    df['month'] = df['job_posted_date'].dt.to_period('M')
    sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    sentiment_trend.plot(kind='line', marker='o', figsize=(15,7), title='Sentiment Trend Over Time')
    plt.ylabel('Number of Job Postings')
    plt.show()

def plot_topic_seniority_heatmap():
    df['topic'] = df['clean_text'].apply(lambda x: lda_model.transform(vectorizer.transform([x]))[0].argmax())
    topic_seniority = df.groupby(['seniority_level', 'topic']).size().unstack().fillna(0)
    plt.figure(figsize=(12,8))
    sns.heatmap(topic_seniority, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Topic Distribution by Seniority Level')
    plt.show()

def plot_word_cloud():
    all_words = ' '.join(df['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Job Descriptions')
    plt.show()

def plot_monthly_postings_trend():
    df['job_posted_date'] = pd.to_datetime(df['job_posted_date'], errors='coerce')
    df['month'] = df['job_posted_date'].dt.to_period('M')
    monthly_trend = df.groupby('month').size()
    monthly_trend.plot(kind='line', marker='o', figsize=(15,7), title='Monthly Job Posting Trend', color='green')
    plt.ylabel('Number of Job Postings')
    plt.show()

def plot_region_distribution():
    df['company_address_region'].value_counts().head(10).plot(kind='bar', color='steelblue', figsize=(12,5))
    plt.title('Top 10 Regions by ML Job Postings')
    plt.show()

def plot_company_sentiment_distribution():
    top_companies = df['company_name'].value_counts().head(5).index
    company_sentiment = df[df['company_name'].isin(top_companies)].groupby(['company_name', 'sentiment']).size().unstack().fillna(0)
    company_sentiment.plot(kind='bar', stacked=True, figsize=(12,6), colormap='Pastel2')
    plt.title('Sentiment Distribution Across Top Companies')
    plt.show()

def plot_sentiment_vs_text_length():
    plt.figure(figsize=(10,6))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Text Length vs Sentiment')
    plt.show()

# Dashboard Handler
def dashboard_handler(change):
    clear_output(wait=True)
    display(dashboard_selector)
    if change['new'] == 'Sentiment Distribution':
        plot_sentiment_distribution()
    elif change['new'] == 'Topics Overview':
        plot_top_topics()
    elif change['new'] == 'Sentiment Over Time':
        plot_sentiment_trend()
    elif change['new'] == 'Topic vs Seniority Heatmap':
        plot_topic_seniority_heatmap()
    elif change['new'] == 'Word Cloud':
        plot_word_cloud()
    elif change['new'] == 'Monthly Posting Trend':
        plot_monthly_postings_trend()
    elif change['new'] == 'Regional Distribution':
        plot_region_distribution()
    elif change['new'] == 'Company Sentiment Distribution':
        plot_company_sentiment_distribution()
    elif change['new'] == 'Sentiment vs Text Length':
        plot_sentiment_vs_text_length()

# Dashboard selector
dashboard_selector = widgets.Dropdown(
    options=[
        'Sentiment Distribution', 'Topics Overview', 'Sentiment Over Time',
        'Topic vs Seniority Heatmap', 'Word Cloud',
        'Monthly Posting Trend', 'Regional Distribution',
        'Company Sentiment Distribution', 'Sentiment vs Text Length'
    ],
    description='Select View:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='70%')
)

dashboard_selector.observe(dashboard_handler, names='value')
display(dashboard_selector)
