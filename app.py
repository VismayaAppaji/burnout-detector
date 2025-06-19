import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the model and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Title
st.title("🧠 Student Burnout Detector")
st.write("Predict your burnout level based on your habits and stress levels.")

# Inputs
sleep = st.selectbox("How many hours do you sleep?", ['4-6 hrs', '6-8 hrs', '8-10 hrs'])
study = st.selectbox("Your average study hours per day?", ['2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5'])
stress = st.slider("Stress level (1 - low, 5 - high)", 1, 5, 3)
social = st.slider("Social connection level (1 - poor, 5 - strong)", 1, 5, 3)
isolation = st.slider("Feeling of isolation (1 - low, 5 - high)", 1, 5, 3)
anxiety = st.slider("Anxiety level (1 - low, 5 - high)", 1, 5, 3)
depression = st.slider("Depression level (1 - low, 5 - high)", 1, 5, 3)

# Convert inputs
sleep_map = {'4-6 hrs': 5, '6-8 hrs': 7, '8-10 hrs': 9}
study_map = {'2.5-3.0': 2.75, '3.0-3.5': 3.25, '3.5-4.0': 3.75, '4.0-4.5': 4.25}

features = np.array([
    sleep_map[sleep],
    study_map[study],
    stress,
    social,
    isolation,
    anxiety,
    depression
]).reshape(1, -1)

# Predict
if st.button("Predict Burnout Level"):
    pred = model.predict(features)[0]
    burnout_label = le.inverse_transform([pred])[0]
    
    st.success(f"🩺 Predicted Burnout Level: *{burnout_label}*")

    if burnout_label == 'High':
        st.error("😟 *You're experiencing high burnout.*\n\nTry these:")
        st.markdown("""
        - 🧘‍♀ Take short mental health breaks (walk, meditate)
        - 💬 Talk to a friend or counselor
        - 🛌 Prioritize 7–8 hrs of sleep
        - 📆 Avoid overloading your study schedule
        - 📱 Reduce screen time & social media scrolling
        """)
    elif burnout_label == 'Medium':
         st.warning("😐 *Moderate stress detected.*\n\nMaintain balance with:")
         st.markdown("""
        - 🧩 Include short hobbies or breaks
        - 🏃‍♀ Move your body — even 15 mins helps
        - 🥗 Eat nutritious meals
        - 📝 Use to-do lists to reduce mental load
        """)
    
    else:
        st.success("🎉 *You're balanced and managing well!*")
        st.markdown("""
        - 🙌 Keep up the great sleep and study habits
        - 🧠 Continue practicing mindfulness or journaling
        - 🌈 Share your routine with friends who need help
        """)
        

# Visual: Feature importance
st.subheader("🔍 What affects burnout the most?")

# Get feature importances from trained model
importances = model.feature_importances_
feature_names = ['Sleep', 'Study Hours', 'Stress', 'Social', 'Isolation', 'Anxiety', 'Depression']

fig, ax = plt.subplots()
sns.barplot(x=importances, y=feature_names, ax=ax, palette="coolwarm")
ax.set_title("Feature Importance")
st.pyplot(fig)
st.markdown("📌 Note: This AI tool is a guide. For mental health concerns, please speak to a counselor or trusted adult")
# # Show prediction result
# if st.button("Predict Burnout Level"):
#     pred = model.predict(features)[0]
#     burnout_label = le.inverse_transform([pred])[0]
    
#     st.success(f"🩺 Predicted Burnout Level: *{burnout_label}*")


