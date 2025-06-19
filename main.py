import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("MentalHealthSurvey.csv")

# Map sleep and study hours to numeric
df['sleep_hours'] = df['sleep'].map({
    '4-6 hrs': 5,
    '6-8 hrs': 7,
    '8-10 hrs': 9
})

df['study_hours'] = df['study'].map({
    '2.5-3.0': 2.75,
    '3.0-3.5': 3.25,
    '3.5-4.0': 3.75,
    '4.0-4.5': 4.25
})

# Burnout label logic
def burnout_label(row):
    score = 0
    if row['sleep'] == '4-6 hrs':
        score += 1
    if row['study'] in ['3.5-4.0', '4.0-4.5']:
        score += 1
    if row['stress'] >= 4:
        score += 1
    if row['depression'] >= 3:
        score += 1
    if row['anxiety'] >= 3:
        score += 1
    if row['isolation'] >= 3:
        score += 1

    if score >= 4:
        return 'High'
    elif score >= 2:
        return 'Medium'
    else:
        return 'Low'
df['burnout_level'] = df.apply(burnout_label, axis=1)

# Select features
features = ['sleep_hours', 'study_hours', 'stress', 'social', 'isolation', 'anxiety', 'depression']
X = df[features]
y = df['burnout_level']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and encoder saved successfully!")
