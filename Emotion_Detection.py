import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import numpy as np

# --- 1. Sample Expanded Dataset ---
data = {
    'text': [
        # Happy
        "I'm bursting with joy today!",
        "I'm on top of the world!",
        "Today just feels perfect!",
        "Yay! I got the job!",
        "That made me smile so much!",
        "Everything is amazing!",
        "I'm grateful for everything.",
        "I feel so happy and alive!",
        "I'm thrilled to start this journey.",
        "My heart is full of joy!",
        "I just won the game!",
        "We are the champions!",
        "Victory is ours!",
        "Celebrate this amazing win!",
        "Hurray!! we have won",
        "Sunshine makes me feel amazing.",
        "I can't stop smiling today!",
        "Everything is going my way.",
        "This is the best day ever!",
        "Nothing can ruin my good mood today!",
        "Just got great news!",
        "My heart feels so light and joyful.",
        "What a fantastic way to start the day!",
        "Today has been full of blessings.",
        "I love spending time with my family.",
        "Such a peaceful and beautiful moment.",
        "This surprise made my whole week!",

        # Sad
        "This is the worst day ever.",
        "I'm so sad and depressed.",
        "Nothing makes me happy anymore.",
        "I’m crying all night.",
        "I'm heartbroken and lonely.",
        "I can't stop the tears.",
        "I don't want to talk to anyone.",
        "I'm not okay today.",
        "I feel broken and empty inside.",
        "I can't stop crying.",
        "I feel like I'm not good enough.",
        "Tears keep falling from my eyes.",
        "It's hard to even get out of bed.",
        "If you cry, I feel Sad",
        "I feel completely hopeless right now.",
        "Loneliness is eating me up inside.",
        "It’s like a dark cloud follows me everywhere.",
        "I can’t handle this pain anymore.",
        "I miss the old days when I was happy.",
        "Nothing ever seems to get better.",
        "I feel invisible to everyone around me.",
        "This sadness won’t go away.",
        "I’m tired of pretending I’m okay.",
        "Everything feels meaningless right now.",

        # Angry
        "I am so angry right now!",
        "Why did you do this? I'm furious!",
        "You make me so mad!",
        "Stop yelling at me!",
        "I feel like punching the wall!",
        "I'm furious and can't hold it in.",
        "This makes my blood boil!",
        "I'm enraged beyond words.",
        "I want to scream at everyone!",
        "I can't tolerate this anymore!",
        "You always ruin everything!",
        "I’m tired of being disrespected.",
        "This behavior makes me furious!",
        "I’m done with all this nonsense!",
        "Every word you say makes me angrier.",
        "Don’t test my patience right now.",
        "I can't believe you betrayed me!",
        "I feel like I’ve been lied to.",
        "I won’t stay quiet anymore!",
        "You crossed the line this time.",
        "Why does no one listen to me?!",
        "I hate being ignored like this.",
        "You’re pushing me to the edge!",
        "Don't you dare !!!",
    ],
    'label': [
        "Happy"] * 27 + ["Sad"] * 24 + ["Angry"] * 24
}
df = pd.DataFrame(data)

# --- 2. Preprocessing ---
df['clean_text'] = df['text'].apply(lambda x: x.lower())
df['clean_text'] = df['clean_text'].apply(nfx.remove_userhandles)
df['clean_text'] = df['clean_text'].apply(nfx.remove_punctuations)

# --- 3. Feature Extraction ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Define Individual Models ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svc_model = SVC(probability=True, kernel='linear', random_state=42)

# --- 6. Voting Classifier (Ensemble) ---
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svc', svc_model)
    ],
    voting='soft'  # use predicted probabilities
)
ensemble_model.fit(X_train, y_train)

# --- 7. Evaluation ---
def evaluate():
    y_pred = ensemble_model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

# --- 8. Save Model and Vectorizer ---
joblib.dump(ensemble_model, "ensemble_emotion_model.pkl")
joblib.dump(vectorizer, "ensemble_vectorizer.pkl")

# --- 9. Prediction Function ---
def predict_emotion(text):
    clean_text = text.lower()
    clean_text = nfx.remove_userhandles(clean_text)
    clean_text = nfx.remove_punctuations(clean_text)
    vect = vectorizer.transform([clean_text])
    return ensemble_model.predict(vect)[0]

# --- 10. Use it ---
if __name__ == "__main__":
    user_input = input("Enter your text: ")
    result = predict_emotion(user_input)
    print("Based on your input, your emotion is:", result)
    print("\nModel Evaluation:")
    evaluate()
