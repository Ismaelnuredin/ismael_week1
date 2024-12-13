# Exploratory Data Analysis (EDA) on Financial News and Stock Price Integration Dataset

This repository contains an Exploratory Data Analysis (EDA) on a **Financial News and Stock Price Integration Dataset**. The analysis includes descriptive statistics, sentiment analysis, topic modeling, and publication trends. The goal is to gain insights from the headlines, understand publication patterns, and identify trends in the financial news articles.

## Contents

- **Descriptive Statistics**: Analyze the headline lengths and article counts per publisher.
- **Publication Dates Analysis**: Identify trends in article publication by day of the week.
- **Publishing Times Analysis**: Explore when articles are published (by hour of the day).
- **Sentiment Analysis**: Classify headlines into Positive, Negative, or Neutral sentiments.
- **Topic Modeling (LDA)**: Identify underlying topics in the headlines using Latent Dirichlet Allocation (LDA).
- **Publisher Analysis**: Analyze which publishers contribute the most articles and extract domain names.

## Requirements

Before running the code, ensure that you have the following Python libraries installed:

- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`

Dataset
The dataset used for this analysis contains financial news articles integrated with stock price data with the following columns:

headline: The headline of the article.
url: The URL of the article.
publisher: The publisher of the article.
date: The date and time of publication.
stock: The associated stock symbol.

**Steps for Running the Analysis**
,Clone or download the repository.
b,Place your dataset file in the appropriate directory.
c,Run the script (eda_analysis.py) in your Python environment or Jupyter notebook.
The script performs the following steps:

Load and clean the dataset.
Perform descriptive statistics on headline lengths and publisher activity.
Analyze publication dates and times.
Perform sentiment analysis on the headlines using VADER.
Use Latent Dirichlet Allocation (LDA) to extract topics from the headlines.
Analyze publisher contributions and extract domains from publisher email addresses.
Output
Descriptive statistics of headline lengths.
Bar plots visualizing the number of articles published by day of the week and hour of the day.
Sentiment distribution showing counts of Positive, Negative, and Neutral headlines.
Topic modeling output displaying top words for each identified topic.
Publisher activity analysis with a count of articles per publisher.






License
This project is licensed under the MIT License - see the LICENSE file for details.
