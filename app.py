import streamlit as st
import pickle
import numpy as np

# Try importing tensorflow and provide a helpful error in Streamlit if it's missing.
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    # Keep the error object available for display and stop execution in Streamlit.
    TENSORFLOW_IMPORT_ERROR = e
    load_model = None
    pad_sequences = None

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    # Always load tokenizer and max_len since they don't require TensorFlow.
    model_obj = None
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    # Load the Keras model only if tensorflow was imported successfully.
    if load_model is not None:
        try:
            model_obj = load_model("lstm_model.h5")
        except Exception as e:
            # If loading the model failed even though tensorflow imported, capture for display.
            globals()["TENSORFLOW_IMPORT_ERROR"] = e
            model_obj = None

    return model_obj, tokenizer, max_len

try:
    model, tokenizer, max_len = load_resources()
except RuntimeError as e:
    # If TensorFlow is missing or failed to import, surface a friendly message in the UI
    model = tokenizer = max_len = None
    IMPORT_ERROR = globals().get("TENSORFLOW_IMPORT_ERROR") or e

# ------------------------------
# Prediction functions
# ------------------------------
def predict_next_word(text):
    """Predict the next single word given input text."""
    # If the model is available, use it.
    if model is not None:
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return ""

    # Fallback heuristic when TensorFlow/model isn't available: return the most frequent word
    # from the tokenizer (a simple demo mode). If tokenizer doesn't have counts, return a common word.
    try:
        if hasattr(tokenizer, "word_counts") and tokenizer.word_counts:
            # tokenizer.word_counts is an ordered dict of word -> count
            # pick the word with the highest count
            top_word = max(tokenizer.word_counts.items(), key=lambda kv: kv[1])[0]
            return top_word
    except Exception:
        pass
    return "the"

def generate_sentence(text, num_words=5):
    """Generate a complete sentence by predicting multiple words."""
    if model is None:
        return "Demo mode: model not available for sentence generation."
    
    current_text = text.strip()
    generated_words = []
    
    for _ in range(num_words):
        # Predict next word
        next_word = predict_next_word(current_text)
        
        if not next_word or next_word == "":
            break
        
        generated_words.append(next_word)
        current_text = current_text + " " + next_word
    
    generated_sentence = text.strip() + " " + " ".join(generated_words)
    return generated_sentence

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word or generate a complete sentence**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# Prediction mode selection
col1, col2 = st.columns(2)
with col1:
    prediction_mode = st.radio("Select mode:", ("Single Word", "Generate Sentence"), horizontal=True)

with col2:
    if prediction_mode == "Generate Sentence":
        num_words = st.slider("Number of words to generate:", min_value=1, max_value=20, value=5)

if model is None:
    st.warning("TensorFlow or the trained model is not available. The app will run in demo mode using a simple heuristic predictor.")
    st.markdown("**Quick options to fix this:**")
    st.markdown("- Create a virtual environment and install dependencies:\n  `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`")
    st.markdown("- Build and run the included Docker image (recommended if you can't install packages system-wide). See the README for commands.")
    st.markdown("- Run `python check_imports.py` to see which imports are failing.")
    if IMPORT_ERROR is not None:
        st.exception(IMPORT_ERROR)

if st.button("🚀 Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        if prediction_mode == "Single Word":
            next_word = predict_next_word(user_input)
            if model is None:
                st.info("(demo mode — heuristic prediction)")
            st.success(f"**Predicted Next Word:** {next_word}")
        else:  # Generate Sentence
            generated_text = generate_sentence(user_input, num_words=num_words)
            if model is None:
                st.info("(demo mode — generation not available)")
            st.success(f"**Generated Sentence:** {generated_text}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")