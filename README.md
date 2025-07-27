# Toxicity Detection App

A machine learning-based web application that classifies text as toxic or non-toxic using Natural Language Processing techniques. The project includes both a Streamlit web interface and a FastAPI backend for text toxicity detection.

## 🎯 Project Overview

This project implements a binary text classification system to detect toxic content in text messages/tweets. It uses TF-IDF vectorization for feature extraction and a Multinomial Naive Bayes classifier for prediction, achieving a ROC-AUC score of **96.59%**.

## 📁 Project Structure

```
toxicity-detection/
│
├── Toxicity_Classifier.ipynb    # Jupyter notebook with model training
├── app.py                       # Streamlit web application
├── api.py                       # FastAPI backend service
├── requirements.txt             # Python dependencies
├── tf_idf.pkt                  # Trained TF-IDF vectorizer (pickle file)
├── toxicity_model.pkt          # Trained Naive Bayes model (pickle file)
└── README.md                   # Project documentation
```

## 🚀 Features

- **Text Preprocessing**: Advanced text cleaning with lemmatization and POS tagging
- **TF-IDF Vectorization**: Feature extraction using Term Frequency-Inverse Document Frequency
- **Machine Learning Model**: Multinomial Naive Bayes classifier
- **Web Interface**: User-friendly Streamlit application
- **API Endpoint**: FastAPI backend for integration with other applications
- **High Accuracy**: Achieved 96.59% ROC-AUC score on test data

## 📊 Dataset

- **Size**: 56,745 text samples
- **Features**: Text content and toxicity labels (0: Non-toxic, 1: Toxic)
- **Distribution**: 
  - Non-toxic: 32,592 samples
  - Toxic: 24,153 samples
- **Source**: Balanced dataset for binary classification

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/toxicity-detection.git
   cd toxicity-detection
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (required for text preprocessing):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('omw-1.4')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```

4. **Download model files:**
   Since the pickle files (`tf_idf.pkt` and `toxicity_model.pkt`) are too large for GitHub, you need to:
   - Train the model using the Jupyter notebook `Toxicity_Classifier.ipynb`, OR
   - Download the pre-trained models from [Google Drive/Dropbox link] (if available)

## 🖥️ Usage

### Streamlit Web Application

Run the Streamlit app for an interactive web interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### FastAPI Backend

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoint

**POST** `/predict`

Request body:
```json
{
  "text": "Your text to analyze"
}
```

Response:
```json
{
  "text": "Your text to analyze",
  "class": "Toxic" or "Non-Toxic"
}
```

### Jupyter Notebook

Open and run the training notebook:

```bash
jupyter notebook Toxicity_Classifier.ipynb
```

## 🔧 Technical Details

### Text Preprocessing Pipeline

1. **Cleaning**: Remove special characters, keep only alphabets and apostrophes
2. **Tokenization**: Split text into individual words
3. **POS Tagging**: Identify parts of speech for each word
4. **Lemmatization**: Convert words to their root forms based on POS tags
5. **Stop Word Removal**: Filter out common English stop words

### Model Architecture

- **Feature Extraction**: TF-IDF Vectorization
- **Algorithm**: Multinomial Naive Bayes
- **Training Split**: 20% training, 80% testing
- **Performance**: 96.59% ROC-AUC score

### Dependencies

```
fastapi
scikit-learn
uvicorn
numpy
streamlit
pandas
matplotlib
nltk
pickle
```

## 📈 Model Performance

- **ROC-AUC Score**: 96.59%
- **Algorithm**: Multinomial Naive Bayes
- **Feature Engineering**: TF-IDF with stop word removal
- **Preprocessing**: Advanced text cleaning with lemmatization

## 🔮 Example Usage

```python
# Example toxic text
input_text = "I hate you moron"
# Prediction: Toxic (probability: ~60%)

# Example non-toxic text  
input_text = "Have a great day!"
# Prediction: Non-Toxic
```

⭐ **Star this repository if you found it helpful!**
