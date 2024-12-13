import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk

# Download NLTK resources (for text preprocessing)
nltk.download('stopwords')

# Load the dataset
def load_data(file_path):
    """Load the dataset from a CSV file"""
    return pd.read_csv(file_path)

# 1. Descriptive Statistics
def plot_headline_length_distribution(df):
    df['headline_length'] = df['headline'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['headline_length'], kde=True)
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Headline Length')
    plt.ylabel('Frequency')
    plt.savefig('results/headline_length_distribution.png')
    plt.show()

def plot_articles_per_publisher(df):
    publisher_counts = df['publisher'].value_counts()
    plt.figure(figsize=(10, 6))
    publisher_counts.head(20).plot(kind='barh', color='lightblue')
    plt.title('Top 20 Publishers by Article Count')
    plt.xlabel('Number of Articles')
    plt.ylabel('Publisher')
    plt.savefig('results/top_publishers.png')
    plt.show()

def plot_articles_over_time(df):
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df.groupby('publication_date').size().plot(figsize=(12, 6))
    plt.title('Article Frequency Over Time')
    plt.xlabel('Publication Date')
    plt.ylabel('Number of Articles')
    plt.savefig('results/articles_over_time.png')
    plt.show()

# 2. Text Analysis
def plot_sentiment_distribution(df):
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], kde=True)
    plt.title('Sentiment Distribution of Headlines')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.savefig('results/sentiment_distribution.png')
    plt.show()

def generate_wordcloud(df):
    stopwords = nltk.corpus.stopwords.words('english')
    text = ' '.join(df['headline'])
    wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Wordcloud of Headlines')
    plt.savefig('results/wordcloud_headlines.png')
    plt.show()

def topic_modeling(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['headline'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = {f"Topic {idx + 1}": [terms[i] for i in topic.argsort()[-10:]] for idx, topic in enumerate(lda.components_)}
    return topics

# 3. Time Series Analysis
def plot_articles_by_month(df):
    df['year'] = df['publication_date'].dt.year
    df['month'] = df['publication_date'].dt.month
    monthly_articles = df.groupby(['year', 'month']).size().reset_index(name='count')
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=pd.to_datetime(monthly_articles[['year', 'month']].assign(day=1)), y=monthly_articles['count'])
    plt.title('Articles Published by Month')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.savefig('results/articles_by_month.png')
    plt.show()

# 4. Publisher Analysis
def plot_top_publishers(df):
    publisher_counts = df['publisher'].value_counts()
    plt.figure(figsize=(10, 6))
    publisher_counts.head(20).plot(kind='barh', color='lightgreen')
    plt.title('Top 20 Publishers')
    plt.xlabel('Article Count')
    plt.ylabel('Publisher')
    plt.savefig('results/top_publishers_by_count.png')
    plt.show()

def analyze_publisher_domains(df):
    df['publisher_domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'No email')
    domain_counts = df['publisher_domain'].value_counts()
    plt.figure(figsize=(10, 6))
    domain_counts.head(20).plot(kind='barh', color='lightcoral')
    plt.title('Top 20 Domains by Publisher Frequency')
    plt.xlabel('Article Count')
    plt.ylabel('Domain')
    plt.savefig('results/top_domains_by_count.png')
    plt.show()
    return domain_counts

# Main Function to Perform All Tasks
def main(file_path):
    df = load_data(file_path)
    plot_headline_length_distribution(df)
    plot_articles_per_publisher(df)
    plot_articles_over_time(df)
    plot_sentiment_distribution(df)
    generate_wordcloud(df)
    topics = topic_modeling(df)
    print("Topics Identified by LDA:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")
    plot_articles_by_month(df)
    plot_top_publishers(df)
    domain_counts = analyze_publisher_domains(df)
    print("Top Publisher Domains:")
    print(domain_counts.head(10))

if __name__ == '__main__':
    main(r'C:\Users\Gaming 15\Desktop\week1_data\raw_analyst_ratings\news_data.csv')
