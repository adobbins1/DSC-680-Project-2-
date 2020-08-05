# Austin Dobbins
# DSC 680

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing Airline Tweets Data Set
df = pd.read_csv(r'C:\Users\austi\OneDrive\Desktop\Tweets.csv')
print(df.head())

# Converting Data Type of Date Features
df['tweet_created'] = pd.to_datetime(df['tweet_created'])
df['date_created'] = df['tweet_created'].dt.date

# Creating Data Frame Containing the Counts of Each Negative, Neutral, and Positive Tweet For Each Airline
# Sorted by Date
d = df.groupby(['date_created', 'airline'])
d = d.airline_sentiment.value_counts()
d.unstack()
print(d)

# Printing the number of Negative, Neutral, and Positive Tweet
print(df.airline_sentiment.value_counts())

# Plotting the Percentage of Tweets for Each Airline
df.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Percentage of Tweets for Each Airline')
plt.show()

# Plotting Percentage of Positive, Negative, and Neutral Comments
df.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['red', 'yellow', 'green'])
plt.title('Percentage of Positive, Negative, and Neutral Comments')
plt.ylabel('Airline Sentiment')
plt.show()

# Count of Negative Reasons
print(df.negativereason.value_counts())

# Plotting Percentage of Reasons for Negative Comments
df.negativereason.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Percentage of Reasons for Negative Comments')
plt.ylabel('Negative Comment Reason')
plt.show()

# Plotting Counts of Positive, Neutral, and Negative Comments for Each Airline
airlinesentiment = df.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airlinesentiment.plot(kind='bar')
plt.title('Counts of Positive, Neutral, and Negative Comments for Each Airline')
plt.xlabel('Airline')
plt.show()

# Plotting Confidence Level for Positive, Neutral, and Negative Tweets
sns.barplot(x= 'airline_sentiment', y = 'airline_sentiment_confidence', data=df)
plt.title('Confidence Level for Positive, Neutral, and Negative Tweets')
plt.xlabel('Airline Sentiment')
plt.ylabel('Airline Sentiment Confidence')
plt.show()

# Removing Unneeded Characters: 'RT' '@'
words = ' '.join(df['text'])
cleanedwords = " ".join([word for word in words.split()
                        if 'http' not in word
                            and not word.startswith('@')
                            and word != 'RT'
                        ])

# Calculating Frequency of Words In Tweets


def freq(str):
    str = str.split()
    str2 = []
    for i in str:
        if i not in str2:
            str2.append(i)
    for i in range(0, len(str2)):
        if str.count(str2[i]) > 50:
            print('Frequency of', str2[i], 'is :', str.count(str2[i]))


# print(freq(cleanedwords))

# Cleaning the Dataset for Modeling
# Dividing Dataset into Features and Labels
features = df.iloc[:, 10].values
labels = df.iloc[:, 1].values

processed_features = []
for sentence in range(0, len(features)):
    # Remove special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Changing multiple spaces to single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

# Creating "Bag of Words" using the 2500 Most Frequently Occurring Words
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

# Starting Model Creation
# Splitting Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

# Random Forest Classifier
textclassifier = RandomForestClassifier(n_estimators=200)
textclassifier.fit(x_train, y_train)

# Random Forest Prediction
predictions = textclassifier.predict(x_test)

# Random Forest Accuracy Metrics
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

# SVM Classifier
textclassifier2 = SVC(gamma='auto')
textclassifier2.fit(x_train, y_train)

# SVM Prediction
predictions2 = textclassifier2.predict(x_test)

# SVM Accuracy Metrics
print(confusion_matrix(y_test, predictions2))
print(classification_report(y_test, predictions2))
print(accuracy_score(y_test, predictions2))

# Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Logistic Regression Prediction
predictions3 = model.predict(x_test)

# Logistic Regression Accuracy Metrics
print(confusion_matrix(y_test, predictions3))
print(classification_report(y_test, predictions3))
print(accuracy_score(y_test, predictions3))

