from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 🔹 1. Load Data

import pandas as pd

# Assuming the file is in the current working directory or you provide the full path
df = pd.read_csv('Tamil_Music_Dataset.csv')
# 🔹 2. Prepare Input Text
df["Input_Text"] = df.drop(columns=["Artist"]).astype(str).apply(lambda x: " ".join(x), axis=1)

# 🔹 3. Encode Labels
label_encoder = LabelEncoder()
df["Artist_Label"] = label_encoder.fit_transform(df["Artist"])
# Load the saved model
loaded_model = TFBertForSequenceClassification.from_pretrained("./bert_artist_tf")

# Load the tokenizer
loaded_tokenizer = BertTokenizer.from_pretrained("./bert_artist_tf")

# Function to make predictions
def predict_artist(text):
    encoding = loaded_tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="tf")
    logits = loaded_model.predict(dict(encoding)).logits
    predicted_class = np.argmax(logits)
    return label_encoder.inverse_transform([predicted_class])[0]

# Test Prediction

if __name__ =="__main__":
    sample_text = "2000s Song A 30 Male Pop 40 Melody Pop"
    predicted_artist = predict_artist(sample_text)
    print(f"Predicted Artist: {predicted_artist}")
