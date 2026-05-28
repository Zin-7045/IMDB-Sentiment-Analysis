import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. Load the Dataset
# ==========================================
print("Loading dataset...")
# Make sure 'IMDB Dataset.csv' is in the same folder as this script
df = pd.read_csv('IMDB Dataset.csv')

# We only need a subset for this assignment to train faster (e.g., 10,000 reviews)
# The Kaggle dataset has 'review' and 'sentiment' columns
df = df.sample(10000, random_state=42) 

# Convert sentiments from text ('positive'/'negative') to numbers (1/0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print(f"Loaded {len(df)} reviews.")

# ==========================================
# 2. Text Preprocessing (Cleaning)
# ==========================================
print("Cleaning text data...")

def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'<br />', ' ', text) # Remove HTML line breaks common in IMDb
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# Apply the cleaning function to the review column
df['cleaned_review'] = df['review'].apply(clean_text)

# Split data into Training (80%) and Testing (20%) sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
)

# ==========================================
# 3. Tokenization & Numerical Representation
# ==========================================
print("Converting text to numbers...")

# We will keep the top 10,000 most frequent words to keep things efficient
max_words = 10000
# We will limit each review to exactly 200 words
max_length = 200 

# Create the Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# Teach the tokenizer the vocabulary from our training data
tokenizer.fit_on_texts(X_train_text)

# Convert the text sentences into lists of numbers
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# ==========================================
# 4. Padding Sequences
# ==========================================
# Deep learning models need inputs to be the exact same size. 
# If a review is short, we add 0s (padding). If it's long, we cut it off (truncating).
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

print("Preprocessing Complete!")
print(f"Shape of training data: {X_train_padded.shape}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# ==========================================
# 5. Build the Simple RNN Model
# ==========================================
print("\n--- Building Simple RNN Model ---")

# Initialize a sequential model (meaning layers go one after another)
rnn_model = Sequential()

# Layer 1: The Embedding Layer
# This turns our word numbers into dense vectors of fixed size (128 dimensions)
# Think of it as mapping words into a mathematical space where similar words are closer together
rnn_model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_length))

# Layer 2: The Simple RNN Layer
# This layer reads the sequences word-by-word. We give it 64 hidden units (memory cells)
rnn_model.add(SimpleRNN(64))

# Layer 3: The Output Layer
# A single neuron with a 'sigmoid' activation function. 
# It outputs a probability between 0 and 1 (closer to 1 = positive review, closer to 0 = negative)
rnn_model.add(Dense(1, activation='sigmoid'))

# Compile the model (telling it how to learn and measure its mistakes)
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Show a summary of the model architecture
rnn_model.summary()

# ==========================================
# 6. Train the Simple RNN Model
# ==========================================
print("\n--- Training Simple RNN Model ---")
# We will train for 5 "epochs" (full passes through the dataset) to save time.
# We also pass in the test data so we can see how well it performs on unseen data after every epoch.
rnn_history = rnn_model.fit(
    X_train_padded, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_padded, y_test)
)

# Evaluate final accuracy on the test set
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_padded, y_test)
print(f"\nSimple RNN Final Test Accuracy: {rnn_accuracy * 100:.2f}%")
from tensorflow.keras.layers import LSTM

# ==========================================
# 7. Build the LSTM Model
# ==========================================
print("\n--- Building LSTM Model ---")

lstm_model = Sequential()
# The exact same embedding layer
lstm_model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_length))

# THE UPGRADE: We swap SimpleRNN for LSTM
lstm_model.add(LSTM(64))

# The exact same output layer
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

# ==========================================
# 8. Train the LSTM Model
# ==========================================
print("\n--- Training LSTM Model ---")
lstm_history = lstm_model.fit(
    X_train_padded, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_padded, y_test)
)

# Evaluate and compare!
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_padded, y_test)
print("\n" + "="*40)
print(f"Simple RNN Final Accuracy: {rnn_accuracy * 100:.2f}%")
print(f"LSTM Final Accuracy:       {lstm_accuracy * 100:.2f}%")
print("="*40 + "\n")
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# ==========================================
# 9. Build the Transformer Model
# ==========================================
print("\n--- Building Transformer Model ---")

# We define the input shape (200 words)
inputs = layers.Input(shape=(max_length,))

# Embedding layer (same as before)
embedding_layer = layers.Embedding(input_dim=max_words, output_dim=128)(inputs)

# THE UPGRADE: Multi-Head Attention (The core of a Transformer)
# It looks at all words simultaneously to find relationships
attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=128)(embedding_layer, embedding_layer)
attention_output = layers.Dropout(0.1)(attention_output)
out1 = layers.LayerNormalization(epsilon=1e-6)(embedding_layer + attention_output)

# Feed Forward Network
ffn_output = layers.Dense(128, activation="relu")(out1)
ffn_output = layers.Dropout(0.1)(ffn_output)
out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Pool the outputs down to a single prediction
x = layers.GlobalAveragePooling1D()(out2)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

# Compile the Transformer model
transformer_model = Model(inputs=inputs, outputs=outputs)
transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
transformer_model.summary()

# ==========================================
# 10. Train the Transformer Model
# ==========================================
print("\n--- Training Transformer Model ---")
transformer_history = transformer_model.fit(
    X_train_padded, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_padded, y_test)
)

# Evaluate and print the final showdown!
transformer_loss, transformer_accuracy = transformer_model.evaluate(X_test_padded, y_test)

print("\n" + "="*50)
print("          FINAL MODEL SHOWDOWN          ")
print("="*50)
print(f"1. Simple RNN Accuracy : {rnn_accuracy * 100:.2f}%")
print(f"2. LSTM Accuracy       : {lstm_accuracy * 100:.2f}%")
print(f"3. Transformer Accuracy: {transformer_accuracy * 100:.2f}%")
print("="*50 + "\n")
import matplotlib.pyplot as plt

# ==========================================
# 11. Plotting the Results
# ==========================================
print("\n--- Generating Comparison Graph ---")

# Define the labels and the data we just calculated
model_names = ['Simple RNN', 'LSTM', 'Transformer']
accuracies = [rnn_accuracy * 100, lstm_accuracy * 100, transformer_accuracy * 100]

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, accuracies, color=['#FF9999', '#66B2FF', '#99FF99'])

# Add titles and labels
plt.title('Deep NLP Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model Architecture', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.ylim(0, 100) # Force the Y-axis to go from 0 to 100

# Add the exact percentage text on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

# Save the graph as an image file in your folder!
plt.savefig('model_comparison_chart.png')
print("Graph saved as 'model_comparison_chart.png'!")

# Display the graph on your screen
plt.show()