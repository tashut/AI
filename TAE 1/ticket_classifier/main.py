import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

# Download stopwords
nltk.download('stopwords')

# Dataset
data = pd.DataFrame({
    'text': [
        'Email not working', 'Outlook not opening',
        'Laptop not starting', 'Keyboard not working',
        'WiFi not connecting', 'Internet slow',
        'VPN connection failed', 'Network disconnecting',
        'Software installation error', 'App crashing',
        'Mouse not detected', 'Screen flickering'
    ],
    'label': [
        'Software', 'Software',
        'Hardware', 'Hardware',
        'Network', 'Network',
        'Network', 'Network',
        'Software', 'Software',
        'Hardware', 'Hardware'
    ]
})

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# Prediction function
def predict_ticket():
    user_input = entry.get()

    if user_input.strip() == "":
        result_label.config(text="⚠ Please enter a ticket!", fg="red")
        return

    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)

    result_label.config(
        text=f"✔ Category: {prediction[0]}",
        fg="#00ffcc"
    )

    # Clear input after prediction
    entry.delete(0, tk.END)

# GUI setup
root = tk.Tk()
root.title("IT Helpdesk AI")
root.geometry("500x350")
root.configure(bg="#1e1e2f")

# Title
tk.Label(
    root,
    text="IT Helpdesk Classifier",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#1e1e2f"
).pack(pady=15)

# Input label
tk.Label(
    root,
    text="Enter your issue:",
    font=("Arial", 12),
    fg="#cccccc",
    bg="#1e1e2f"
).pack()

# Entry box
entry = tk.Entry(
    root,
    width=40,
    font=("Arial", 12),
    bg="#2e2e3f",
    fg="white",
    insertbackground="white"
)
entry.pack(pady=10)

# Predict button
tk.Button(
    root,
    text="Predict",
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5,
    command=predict_ticket
).pack(pady=10)

# Result label
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold"),
    bg="#1e1e2f"
)
result_label.pack(pady=20)

# Footer
tk.Label(
    root,
    text="AI-based Ticket Classifier",
    font=("Arial", 9),
    fg="gray",
    bg="#1e1e2f"
).pack(side="bottom", pady=10)

# Run app
root.mainloop()