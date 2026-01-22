import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load in the match level dataset
df = pd.read_csv("match_features.csv") 

# Drop the rows with missing values (first 4/5 games where a teams form is not yet calculated)
df = df.dropna()
pd.set_option('display.max_columns', None)

# Sort the feature columns by date
feature_columns = ['HomeForm', 'AwayForm', 'HomeGoalsLast5', 'AwayGoalsLast5', 'HomeWinsLast5', 'AwayWinsLast5', 'HomeGoalsConcededLast5', 'AwayGoalsConcededLast5']
df = df.sort_values('Date')

# X = input matrix (features)
# Y = output labels (what we want to predict)
X = df[feature_columns]
y = df['Result'] # W / D / L from home teams perspective

# Split without shuffling to train on the past and predict the future games
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialise random forest model; random_state ensures reproducibility
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
accuracy = rfc.score(X_test, y_test)

print(df.head())
print("Model Accuracy: ", accuracy)

# Shows how features relate to one another
plt.figure(figsize=(10,8))
sns.heatmap(
    df[feature_columns].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title('Feature Correlation')
plt.show() 

# Compares the actual results vs predicted results
plt.figure(figsize=(8,6))
sns.countplot(
    x=y_test,        # Actual results
    hue=y_pred,      # Predictions
    palette='Set2'
)
plt.title('Predicted vs Actual Results')
plt.xlabel('Actual Result')
plt.ylabel('Count')
plt.show()

# Shows which features the forest relied on most
importances = pd.Series(
    rfc.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)