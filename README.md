# Audio Emotion Recognition System

An ML-based system for recognizing human emotions from speech audio samples, using acoustic feature extraction and classical machine learning approaches.

## Project Overview

This project implements an audio-based emotion recognition system capable of classifying emotional states from speech samples. The system follows a complete machine learning pipeline including:

- Data acquisition from the CREMA-D dataset
- Audio preprocessing (noise reduction, normalization)
- Feature extraction (MFCC, spectral features, prosodic features)
- Model training and evaluation (Random Forest, XGBoost)
- Statistical analysis and interpretation

## Installation

### Prerequisites

- Python 3.8+
- Required libraries (install via pip):

```bash
pip install -r requirements.txt
```

### Optional GPU Support

For GPU acceleration:

```bash
# For XGBoost GPU support
pip install cupy-cuda11x  # Replace 11x with your CUDA version

# For PyTorch GPU support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Project Structure

```
├── config.yaml           # Configuration settings
├── data/                 # Data directory
│   ├── raw/              # Raw audio files
│   ├── processed/        # Processed audio segments
│   └── features/         # Extracted feature datasets
├── docs/                 # Documentation
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for analysis
├── results/              # Results and visualizations
│   ├── analysis/         # Analysis outputs
│   ├── feature_importance/ # Feature importance visualizations  
│   └── statistical_analysis/ # Statistical test results
├── src/                  # Source code
│   ├── cremad_loader.py  # Data loading utilities
│   ├── preprocessing.py  # Audio preprocessing
│   ├── features.py       # Feature extraction
│   ├── models.py         # ML model implementation
│   ├── models_gpu.py     # GPU-accelerated models
│   └── statistical_analysis.py  # Statistical testing
└── main.py               # Main execution script
```

## Usage

### Environment Setup

```bash
# Check environment setup
python main.py --check-env

# Verify project structure
python main.py --verify-structure

# Check GPU capabilities (optional)
python main.py --check-gpu
```

### Data Processing

```bash
# Process audio dataset and extract features
python main.py --process-dataset

# Limit number of files to process
python main.py --process-dataset --limit 500
```

### Model Training and Evaluation

```bash
# Cross-validate models
python main.py --cross-validate --output-dir results

# Develop and evaluate models
python main.py --develop-models --output-dir results

# Train models with GPU acceleration (if available)
python main.py --train --use-gpu
```

### Prediction

```bash
# Predict emotion for a single audio file
python main.py --predict --audio-file path/to/audio.wav

# Batch predict emotions for audio files in a directory
python main.py --batch-predict --audio-dir path/to/audio/directory
```

### Analysis

```bash
# Analyze prediction results
python main.py --analyze-predictions --predictions-file results/batch_predictions.csv --output-dir results/analysis

# Analyze feature importance
python main.py --analyze-feature-importance --model-path models/prediction_pipeline.pkl --output-dir results/feature_importance

# Perform statistical analysis on model results
python main.py --statistical-analysis --output-dir results
```

## Model Performance

Based on our statistical analysis, XGBoost significantly outperforms Random Forest on the CREMA-D dataset for emotion recognition:

- XGBoost achieves ~56.5% accuracy (95% CI: 55.8-57.4%)
- Random Forest achieves ~54.1% accuracy (95% CI: 53.3-55.3%)
- Statistical significance: p = 0.012 (paired t-test)

The significant difference indicates that XGBoost is consistently better at recognizing emotions from speech, with particular strength in distinguishing between angry, disgust, and sad emotions.

## Feature Importance

The most important features for emotion classification are:
1. MFCC features (particularly their statistical properties)
2. Spectral contrast features
3. Zero crossing rate and spectral bandwidth

These provide insights into which acoustic characteristics best differentiate emotional speech.

## Limitations and Future Work

Current limitations include:
- Poor discrimination between happy and neutral emotions
- Over-reliance on MFCC features
- Limited temporal modeling of emotion changes

Potential improvements:
- Implement deep learning approaches (CNNs for spectrograms)
- Add specialized features for commonly confused emotions
- Develop hierarchical classifiers for similar emotions
- Implement data augmentation for better class balance

See `docs/limitations_and_improvements.md` for a detailed analysis.

## Documentation

- `docs/statistical_analysis_guide.md`: Guide to statistical analysis functionality
- `docs/limitations_and_improvements.md`: Detailed analysis of limitations and potential improvements

## License

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