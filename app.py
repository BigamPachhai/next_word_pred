import streamlit as st
import pickle
import numpy as np
import h5py
import json

# Try importing tensorflow and provide a helpful error in Streamlit if it's missing.
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except Exception as e:
    # Keep the error object available for display and stop execution in Streamlit.
    TENSORFLOW_IMPORT_ERROR = e
    load_model = None
    pad_sequences = None
    HAS_TENSORFLOW = False

# Fallback: Manual tokenization and padding if TensorFlow is unavailable
def fallback_pad_sequences(sequences, maxlen, padding='pre'):
    """Manual padding implementation without TensorFlow."""
    padded = []
    for seq in sequences:
        if padding == 'pre':
            pad_amount = maxlen - len(seq)
            padded.append([0] * pad_amount + seq)
        else:
            pad_amount = maxlen - len(seq)
            padded.append(seq + [0] * pad_amount)
    return np.array(padded)

# Model prediction without TensorFlow (using direct h5 access)
def load_model_from_h5(filepath):
    """Load model architecture and weights from h5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            # For now, we'll just return None and rely on the fallback
            # A full implementation would require reconstructing the model from h5py
            return None
    except Exception as e:
        return None

# Use fallback padding if TensorFlow is not available
if not HAS_TENSORFLOW:
    pad_sequences = fallback_pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    # Always load tokenizer and max_len since they don't require TensorFlow.
    model_obj = None
    tokenizer = None
    max_len = None
    
    # Try to load tokenizer
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        # Silently continue - we have demo mode fallback
        tokenizer = None
    except (ModuleNotFoundError, ImportError) as e:
        # Handle missing Keras/TensorFlow during unpickling
        # This is expected in Python 3.14 environments
        if "keras" in str(e).lower() or "tensorflow" in str(e).lower():
            # This is a known limitation in Python 3.14, silently use demo mode
            tokenizer = None
        else:
            st.warning(f"⚠️ Could not load tokenizer: {str(e)}")
            tokenizer = None
    except Exception as e:
        st.warning(f"⚠️ Could not load tokenizer: {str(e)}")
        tokenizer = None
    
    # Try to load max_len
    try:
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
    except FileNotFoundError:
        # Silently continue
        max_len = None
    except (ModuleNotFoundError, ImportError) as e:
        # Handle missing Keras/TensorFlow during unpickling
        if "keras" in str(e).lower() or "tensorflow" in str(e).lower():
            max_len = None
        else:
            st.warning(f"⚠️ Could not load max_len: {str(e)}")
            max_len = None

    # Load the Keras model only if tensorflow was imported successfully.
    if HAS_TENSORFLOW and load_model is not None:
        try:
            model_obj = load_model("lstm_model.h5")
        except Exception as e:
            # If loading the model failed even though tensorflow imported, capture for display.
            globals()["TENSORFLOW_IMPORT_ERROR"] = e
            model_obj = None
    elif not HAS_TENSORFLOW:
        # Try to load model from h5py if TensorFlow is not available
        model_obj = None

    return model_obj, tokenizer, max_len

try:
    model, tokenizer, max_len = load_resources()
except Exception as e:
    # Catch any other exceptions
    model = None
    tokenizer = None
    max_len = None
    IMPORT_ERROR = str(e)

# ------------------------------
# Prediction functions
# ------------------------------
def predict_next_word(text):
    """Predict the next single word given input text."""
    # If the model and tokenizer are available, use them
    if model is not None and tokenizer is not None and max_len is not None:
        try:
            sequence = tokenizer.texts_to_sequences([text])[0]
            sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

            preds = model.predict(sequence, verbose=0)
            predicted_index = np.argmax(preds)

            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    return word
            return ""
        except Exception as e:
            return ""

    # Fallback heuristic when model/tokenizer isn't available: return the most frequent word
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "word_counts") and tokenizer.word_counts:
                # tokenizer.word_counts is an ordered dict of word -> count
                # pick the word with the highest count
                top_word = max(tokenizer.word_counts.items(), key=lambda kv: kv[1])[0]
                return top_word
        except Exception:
            pass
    
    # Final fallback: return a common word
    return "the"

def generate_sentence(text, num_words=5):
    """Generate a complete sentence by predicting multiple words."""
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

if model is None or tokenizer is None or max_len is None:
    st.warning("⚠️ Pre-trained model/files not available. Running in **demo mode** with heuristic predictions.")
    st.info("""
    **Why?** The Streamlit Cloud environment uses Python 3.14 or model files are missing.
    
    **To use the full model locally:**
    - Use Python 3.12 or 3.13
    - Create a virtual environment: `python -m venv .venv`
    - Activate it: `source .venv/bin/activate` (or `.venv\\Scripts\\activate` on Windows)
    - Install dependencies: `pip install -r requirements.txt`
    - Run locally: `streamlit run app.py`
    """)
    if HAS_TENSORFLOW and 'IMPORT_ERROR' in globals() and IMPORT_ERROR is not None:
        with st.expander("📋 Technical Details"):
            st.code(str(IMPORT_ERROR))

if st.button("🚀 Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        if prediction_mode == "Single Word":
            next_word = predict_next_word(user_input)
            if model is None or tokenizer is None:
                st.info("(demo mode — heuristic prediction)")
            st.success(f"**Predicted Next Word:** {next_word}")
        else:  # Generate Sentence
            generated_text = generate_sentence(user_input, num_words=num_words)
            if model is None or tokenizer is None:
                st.info("(demo mode — heuristic generation)")
            st.success(f"**Generated Sentence:** {generated_text}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")