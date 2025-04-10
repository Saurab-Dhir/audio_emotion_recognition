# Audio-Based Emotion Recognition System

This project implements a machine learning pipeline for recognizing emotions from speech audio data. The system analyzes audio samples to classify them into one of eight emotional states.

## Project Overview

The audio-based emotion recognition system follows a complete machine learning pipeline:

1. **Data Acquisition**: Using the RAVDESS emotional speech dataset
2. **Preprocessing**: Noise reduction, normalization, and segmentation
3. **Feature Extraction**: MFCC, prosodic, and spectral features
4. **Model Development**: Implementation of SVM, Random Forest, and XGBoost models
5. **Evaluation**: Comprehensive performance metrics and analysis

## Requirements

- Python 3.8+
- Libraries: numpy, pandas, scikit-learn, librosa, matplotlib, etc. (see requirements.txt)

## Project Structure

```
audio_emotion_recognition/
├── data/
│   ├── raw/          # For RAVDESS dataset
│   ├── processed/    # For preprocessed audio files
│   └── features/     # For extracted features
├── notebooks/        # Jupyter notebooks for exploration and visualization
├── src/              # Source code
│   ├── preprocessing.py   # Audio preprocessing functions
│   ├── features.py        # Feature extraction pipeline
│   ├── models.py          # ML models implementation
│   ├── evaluation.py      # Evaluation metrics and analysis
│   └── utils.py           # Utility functions
├── models/           # Saved model files
├── results/          # Evaluation results and visualizations
├── docs/             # Documentation
├── main.py           # Main script to run the pipeline
├── config.yaml       # Configuration parameters
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/audio_emotion_recognition.git
cd audio_emotion_recognition
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the RAVDESS Dataset

Download the RAVDESS dataset from [Zenodo](https://zenodo.org/record/1188976) and extract it to the `data/raw/ravdess/` directory.

### 4. Verify Setup

```bash
python main.py --check-env --verify-structure
```

### 5. Explore the Dataset

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_Data_Exploration.ipynb
```

## Pipeline Execution

### Preprocessing

The preprocessing module (`src/preprocessing.py`) handles:
- Noise reduction using spectral gating
- Amplitude normalization
- Silence removal
- Audio segmentation

### Feature Extraction

The feature extraction module (`src/features.py`) extracts:
- MFCC features (including deltas and delta-deltas)
- Prosodic features (pitch, energy, speaking rate)
- Spectral features (centroid, flux, roll-off)

### Model Training and Evaluation

Run the complete pipeline:

```bash
python main.py
```

To run specific parts of the pipeline:

```bash
# Process the dataset and extract features
python main.py --process

# Train a neural network model with GPU acceleration 
python main.py --train --feature-path data/features/cremad_features.pkl

# Evaluate a trained model
python main.py --evaluate --model-path models/emotion_model.pt
```

### Model Development

The system provides comprehensive model development capabilities including:

1. **Feature Selection**: Select the most relevant features using ANOVA F-value or Recursive Feature Elimination
2. **Machine Learning Models**: Train and compare Random Forest and XGBoost classifiers
3. **GPU Acceleration**: Use NVIDIA GPU acceleration for XGBoost models for faster training and inference
4. **Hyperparameter Optimization**: Find optimal hyperparameters for each model type
5. **Model Evaluation**: Compare models using cross-validation and independent test set
6. **Model Selection**: Automatically select the best performing model

To develop and compare traditional machine learning models:

```bash
# Run the model development pipeline with default settings
python main.py --develop-models

# Run with custom settings
python main.py --develop-models --feature-path data/features/cremad_features.pkl --model-dir models/

# Use GPU acceleration (default if GPU is available)
python main.py --develop-models

# Force CPU-only mode (no GPU)
python main.py --develop-models --cpu

# Disable feature selection or hyperparameter tuning for faster development
python main.py --develop-models --no-feature-selection --no-hyperparameter-tuning
```

The model development process will:
- Split data into train/validation/test sets
- Apply feature selection (if enabled)
- Train multiple model types
- Optimize hyperparameters (if enabled)
- Compare model performances
- Select the best model
- Save detailed results and visualizations

### Cross-Validation

To perform k-fold cross-validation and compare model performances:

```bash
# Run 5-fold cross-validation with default settings
python main.py --cross-validate

# Run with custom settings
python main.py --cross-validate --n-folds 10 --feature-path data/features/cremad_features.pkl

# Disable feature selection for faster cross-validation
python main.py --cross-validate --no-feature-selection
```

The cross-validation process will:
- Perform k-fold cross-validation for each model type
- Calculate performance metrics for each fold
- Generate summary statistics (mean and standard deviation)
- Create visualizations comparing model performances
- Save detailed results to the results directory

### Emotion Prediction

Once models are trained, you can use them to predict emotions from new audio files:

```bash
# Predict emotion for a single audio file
python main.py --predict --audio-file path/to/audio.wav

# Predict emotions for a batch of audio files in a directory
python main.py --batch-predict --audio-dir path/to/audio/directory

# Use a specific model for prediction
python main.py --predict --audio-file path/to/audio.wav --model-path models/xgboost_model.pkl
```

The prediction pipeline:
1. Loads the audio file(s)
2. Applies the same preprocessing steps used during training
3. Extracts the same feature set
4. Uses the trained model to classify the emotion
5. For batch prediction, saves results to a CSV file

## Results

The evaluation results including accuracy, precision, recall, F1-score, confusion matrices, and ROC curves will be saved in the `results/` directory.

For model development results, check the `models/` directory for:
- Trained model files (`.pkl`)
- Classification reports (`.csv`)
- Confusion matrix visualizations (`.png`)
- Prediction pipeline for the best model

## License

[MIT License](LICENSE)