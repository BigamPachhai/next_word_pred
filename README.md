Next Word Prediction (LSTM)

This Streamlit app predicts the next word or generates complete sentences using an LSTM model.

Setup

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Launch the app:

   streamlit run app.py

## Features

- **Single Word Prediction**: Predicts the next word based on your input
- **Sentence Generation**: Generates complete sentences by predicting multiple words sequentially
- **Configurable Length**: Choose how many words to generate (1-20)
- **Demo Mode**: Works even without the trained model using heuristic fallback

Notes

- If your system has a GPU and proper CUDA/cuDNN, install `tensorflow` instead of `tensorflow-cpu` in `requirements.txt`.
- `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` must be present in the project root.
