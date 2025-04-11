#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Audio-Based Emotion Recognition System
"""

import os
import argparse
import logging
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import glob
import librosa

from src.utils import load_config, create_directories
from src.cremad_loader import CREMADDataLoader
from src.preprocessing import AudioPreprocessor
from src.features import FeatureExtractor
from models.models_gpu import GPUModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """
    Check if the environment is correctly set up
    """
    import sys
    import numpy
    import pandas
    import matplotlib
    import librosa
    import sklearn
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {numpy.__version__}")
    logger.info(f"Pandas version: {pandas.__version__}")
    logger.info(f"Matplotlib version: {matplotlib.__version__}")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")

def test_cremad_loader(config, limit=5):
    """
    Test the CREMA-D data loader with a small subset
    
    Args:
        config (dict): Configuration dictionary
        limit (int): Number of samples to load
    """
    logger.info("Testing CREMA-D data loader...")
    
    # Initialize data loader
    data_loader = CREMADDataLoader(config_path="config.yaml")
    
    # Load a small subset of the data
    metadata_df, audio_data = data_loader.load_dataset(limit=limit)
    
    if not metadata_df.empty and len(audio_data) > 0:
        logger.info(f"Successfully loaded {len(metadata_df)} samples")
        logger.info(f"Metadata sample:\n{metadata_df.head()}")
        
        y, sr = audio_data[0]
        logger.info(f"Audio sample: Shape={y.shape}, SR={sr}, Duration={len(y)/sr:.2f}s")
        return True
    else:
        logger.warning("No data loaded. Check your dataset path and file format.")
        return False

def verify_project_structure():
    """
    Verify that the project structure is correctly set up
    """
    expected_dirs = [
        "data/raw/cremad",
        "data/processed",
        "data/features",
        "notebooks",
        "src",
        "models",
        "results"
    ]
    
    for dir_path in expected_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} does not exist")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory {dir_path}")
        else:
            logger.info(f"Directory {dir_path} exists")
    
    expected_files = [
        "config.yaml",
        "requirements.txt",
        "src/utils.py",
        "src/cremad_loader.py",
        "src/preprocessing.py",
        "src/features.py"
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
        else:
            logger.info(f"File {file_path} exists")

def process_dataset(limit=None, output_path=None):
    """
    Process the dataset with the complete pipeline
    
    Args:
        limit (int, optional): Number of samples to process
        output_path (str, optional): Path to save processed features
    """
    logger.info("Starting dataset processing pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories(config)
    
    # Initialize components
    data_loader = CREMADDataLoader()
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Load dataset
    logger.info(f"Loading dataset{' (limited to ' + str(limit) + ' samples)' if limit else ''}...")
    metadata_df, audio_data = data_loader.load_dataset(limit=limit, stratify_by='emotion')
    
    if metadata_df.empty or not audio_data:
        logger.error("No data loaded. Aborting processing.")
        return False
    
    logger.info(f"Loaded {len(metadata_df)} samples")
    
    # Preprocess audio
    logger.info("Preprocessing audio...")
    preprocessed_segments = []
    
    for y, sr in tqdm(audio_data, desc="Preprocessing"):
        segments = preprocessor.preprocess_and_segment(y, sr)
        preprocessed_segments.append(segments)
    
    # Extract features
    logger.info("Extracting features...")
    all_features = feature_extractor.extract_features_batch(
        preprocessed_segments, data_loader.sample_rate, compute_statistics=True
    )
    
    # Create feature dataset
    logger.info("Creating feature dataset...")
    feature_df = feature_extractor.create_feature_dataset(metadata_df, all_features)
    
    # Save features
    if output_path is None:
        output_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    logger.info(f"Saving features to {output_path}...")
    feature_extractor.save_features(feature_df, output_path)
    
    logger.info(f"Dataset processing complete. {len(feature_df)} feature vectors saved.")
    return True

def train_models(feature_path=None, model_output=None, use_gpu=True, num_epochs=10):
    """
    Train models on the extracted features
    
    Args:
        feature_path (str, optional): Path to feature dataset
        model_output (str, optional): Path to save trained model
        use_gpu (bool): Whether to use GPU for training
        num_epochs (int): Number of training epochs
    """
    logger.info("Starting model training...")
    
    # Load configuration
    config = load_config()
    
    # Update GPU usage in config
    if 'model' not in config:
        config['model'] = {}
    if 'gpu' not in config['model']:
        config['model']['gpu'] = {}
    config['model']['gpu']['use_gpu'] = use_gpu
    
    # Set default paths if not provided
    if feature_path is None:
        feature_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    if model_output is None:
        model_output = os.path.join(config['paths']['models'], 'emotion_model.pt')
    
    # Check if features exist
    if not os.path.exists(feature_path):
        logger.error(f"Feature file not found: {feature_path}")
        return False
    
    # Load features
    logger.info(f"Loading features from {feature_path}...")
    feature_extractor = FeatureExtractor()
    feature_df = feature_extractor.load_features(feature_path)
    
    # Initialize model trainer
    model_trainer = GPUModelTrainer()
    
    # Prepare data
    logger.info("Preparing data for training...")
    train_loader, test_loader, X_test, y_test = model_trainer.prepare_data(feature_df)
    
    # Train model
    logger.info("Training model...")
    model = model_trainer.train(train_loader, num_epochs=num_epochs)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model_trainer.evaluate(test_loader)
    
    # Save model
    logger.info(f"Saving model to {model_output}...")
    model_trainer.save_model(model_output)
    
    # Save metrics
    metrics_path = os.path.join(config['paths']['results'], 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model training complete. Results saved to {model_output}")
    return True

def evaluate_model(model_path=None, feature_path=None):
    """
    Evaluate a trained model on the test set
    
    Args:
        model_path (str, optional): Path to the trained model
        feature_path (str, optional): Path to feature dataset
    """
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], 'emotion_model.pt')
    
    if feature_path is None:
        feature_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(feature_path):
        logger.error(f"Feature file not found: {feature_path}")
        return False
    
    # Load features
    logger.info(f"Loading features from {feature_path}...")
    feature_extractor = FeatureExtractor()
    feature_df = feature_extractor.load_features(feature_path)
    
    # Initialize model trainer and load model
    model_trainer = GPUModelTrainer()
    model_trainer.load_model(model_path)
    
    # Prepare data
    logger.info("Preparing data for evaluation...")
    _, test_loader, _, _ = model_trainer.prepare_data(feature_df)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model_trainer.evaluate(test_loader)
    
    # Print metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    return True

def develop_models(feature_path=None, model_output_dir=None, use_feature_selection=True, tune_hyperparams=True):
    """
    Develop and evaluate machine learning models
    
    Args:
        feature_path (str, optional): Path to feature dataset
        model_output_dir (str, optional): Directory to save trained models
        use_feature_selection (bool): Whether to use feature selection
        tune_hyperparams (bool): Whether to tune hyperparameters
    """
    logger.info("Starting model development pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if feature_path is None:
        feature_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    if model_output_dir is None:
        model_output_dir = config['paths']['models']
    
    # Initialize model trainer
    from src.models import EmotionModelTrainer
    trainer = EmotionModelTrainer()
    
    # Load feature dataset
    feature_df = trainer.load_feature_dataset(feature_path)
    
    # Split data into train/validation/test sets
    # Use 70% for training, 15% for validation, 15% for test
    target_col = 'emotion'
    test_size = 0.15
    val_size = 0.15 / (1 - test_size)  # adjusted to get 15% of original data
    random_state = config['model'].get('random_state', 42)
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test, feature_names = trainer.prepare_data(
        feature_df, target_col=target_col, test_size=test_size, random_state=random_state
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Apply feature selection if requested
    if use_feature_selection:
        feature_selection_method = config['model'].get('feature_selection', {}).get('method', 'selectk')
        k_features = config['model'].get('feature_selection', {}).get('k_features', 100)
        
        X_train, X_test, selected_features = trainer.select_features(
            X_train, y_train, X_test, method=feature_selection_method, k=k_features
        )
        
        # Apply the same transformation to validation set
        if feature_selection_method == 'selectk':
            selector = SelectKBest(f_classif, k=k_features)
            selector.fit(X_train_val, y_train_val)  # Fit on the combined train+val data
            X_val = selector.transform(X_val)
        elif feature_selection_method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k_features, step=0.1)
            selector.fit(X_train_val, y_train_val)  # Fit on the combined train+val data
            X_val = selector.transform(X_val)
        
        logger.info(f"Selected {len(selected_features)} features using {feature_selection_method}")
    
    # Train models
    models = {}
    
    # Get GPU usage setting from config
    use_gpu = config['model'].get('device', 'cuda').lower() == 'cuda' and config['model'].get('gpu', {}).get('use_gpu', True)
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Train Random Forest model
    logger.info("Training Random Forest model...")
    rf_model = trainer.train_random_forest(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
    models['random_forest'] = rf_model[0] if isinstance(rf_model, tuple) else rf_model
    
    # Train XGBoost model with GPU if available
    logger.info("Training XGBoost model...")
    trainer.use_gpu = use_gpu  # Set the use_gpu flag on the trainer object
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
    models['xgboost'] = xgb_model[0] if isinstance(xgb_model, tuple) else xgb_model
    
    # Evaluate models on validation set
    val_results = {}
    for name, model in models.items():
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        val_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"{name.upper()} validation results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    # Select best model based on validation accuracy
    best_model_name = max(val_results, key=lambda x: val_results[x]['accuracy'])
    best_model = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name.upper()} with validation accuracy {val_results[best_model_name]['accuracy']:.4f}")
    
    # Evaluate best model on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Best model test results: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}")
    
    # Save detailed classification report
    report = classification_report(y_test, y_pred, target_names=trainer.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(model_output_dir, f"{best_model_name}_classification_report.csv")
    report_df.to_csv(report_path)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=trainer.classes_, yticklabels=trainer.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model_name.upper()}')
    cm_path = os.path.join(model_output_dir, f"{best_model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save all models
    for name, model in models.items():
        model_path = os.path.join(model_output_dir, f"{name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} model to {model_path}")
    
    # Create a prediction pipeline and save it
    pipeline = {
        'model': best_model,
        'scaler': trainer.scaler,
        'label_encoder': trainer.label_encoder,
        'selected_features': selected_features if use_feature_selection else list(range(X_train.shape[1])),
        'classes': trainer.classes_
    }
    
    pipeline_path = os.path.join(model_output_dir, 'prediction_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved prediction pipeline to {pipeline_path}")
    
    return best_model_name, val_results, test_accuracy

def cross_validate_models(feature_path=None, n_folds=5, output_dir=None, use_feature_selection=True):
    """
    Perform cross-validation to evaluate model performance
    
    Args:
        feature_path (str, optional): Path to feature dataset
        n_folds (int): Number of cross-validation folds
        output_dir (str, optional): Directory to save results
        use_feature_selection (bool): Whether to use feature selection
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if feature_path is None:
        feature_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    if output_dir is None:
        output_dir = config['paths']['results']
    
    # Initialize model trainer
    from src.models import EmotionModelTrainer
    trainer = EmotionModelTrainer()
    
    # Load feature dataset
    feature_df = trainer.load_feature_dataset(feature_path)
    
    # Extract features and target
    target_col = 'emotion'
    X = np.vstack(feature_df['feature_vector'].values)
    y = feature_df[target_col].values
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Initialize StratifiedKFold
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['model'].get('random_state', 42))
    
    # Initialize results dictionaries
    cv_results = {
        'random_forest': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'xgboost': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    # Perform cross-validation
    fold_idx = 1
    scaler = StandardScaler()
    
    for train_idx, test_idx in kfold.split(X, y_encoded):
        logger.info(f"Processing fold {fold_idx}/{n_folds}...")
        
        # Split data into train and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Scale features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply feature selection if requested
        if use_feature_selection:
            feature_selection_method = config['model'].get('feature_selection', {}).get('method', 'selectk')
            k_features = config['model'].get('feature_selection', {}).get('k_features', 100)
            
            if feature_selection_method == 'selectk':
                selector = SelectKBest(f_classif, k=k_features)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
            elif feature_selection_method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator, n_features_to_select=k_features, step=0.1)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
        
        # Get GPU usage setting from config
        use_gpu = config['model'].get('device', 'cuda').lower() == 'cuda' and config['model'].get('gpu', {}).get('use_gpu', True)
        
        # Split training data to create a validation set for model training
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=config['model'].get('random_state', 42), stratify=y_train
        )
        
        # Random Forest
        rf_model = trainer.train_random_forest(X_train_split, y_train_split, X_val, y_val, tune_hyperparams=False)
        y_pred = rf_model[0].predict(X_test) if isinstance(rf_model, tuple) else rf_model.predict(X_test)
        
        cv_results['random_forest']['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_results['random_forest']['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        cv_results['random_forest']['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        cv_results['random_forest']['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        
        # XGBoost with GPU if available
        trainer.use_gpu = use_gpu  # Set the use_gpu flag on the trainer object
        xgb_model = trainer.train_xgboost(X_train_split, y_train_split, X_val, y_val, tune_hyperparams=False)
        y_pred = xgb_model[0].predict(X_test) if isinstance(xgb_model, tuple) else xgb_model.predict(X_test)
        
        cv_results['xgboost']['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_results['xgboost']['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        cv_results['xgboost']['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        cv_results['xgboost']['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        
        fold_idx += 1
    
    # Calculate mean and std of metrics
    results_summary = {}
    for model_name, metrics in cv_results.items():
        results_summary[model_name] = {}
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            results_summary[model_name][f'{metric_name}_mean'] = mean_value
            results_summary[model_name][f'{metric_name}_std'] = std_value
            logger.info(f"{model_name.upper()} {metric_name}: {mean_value:.4f} ± {std_value:.4f}")
    
    # Create boxplot of accuracies
    plt.figure(figsize=(10, 6))
    accuracies = [cv_results['random_forest']['accuracy'], 
                 cv_results['xgboost']['accuracy']]
    
    plt.boxplot(accuracies, labels=['Random Forest', 'XGBoost'])
    plt.title('Cross-Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    boxplot_path = os.path.join(output_dir, 'cv_accuracy_comparison.png')
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Saved accuracy comparison plot to {boxplot_path}")
    
    # Save results as CSV
    results_df = pd.DataFrame(results_summary).T
    results_df.index.name = 'model'
    
    csv_path = os.path.join(output_dir, 'cross_validation_results.csv')
    results_df.to_csv(csv_path)
    logger.info(f"Saved cross-validation results to {csv_path}")
    
    # Save raw cross-validation results for statistical analysis
    cv_results_path = os.path.join(output_dir, 'cv_results.pkl')
    with open(cv_results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    logger.info(f"Saved raw cross-validation results to {cv_results_path}")
    
    # Determine best model
    best_model = max(results_summary, key=lambda x: results_summary[x]['accuracy_mean'])
    logger.info(f"Best model based on cross-validation: {best_model.upper()} "
               f"with accuracy {results_summary[best_model]['accuracy_mean']:.4f} "
               f"± {results_summary[best_model]['accuracy_std']:.4f}")
    
    return best_model, results_summary

def predict_emotion(audio_file=None, model_path=None, config_path="config.yaml"):
    """
    Predict emotion for an audio file
    
    Args:
        audio_file (str): Path to audio file
        model_path (str): Path to model file
        config_path (str): Path to configuration file
    
    Returns:
        str: Predicted emotion
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], 'prediction_pipeline.pkl')
    
    if audio_file is None:
        logger.error("No audio file provided")
        return None
    
    logger.info(f"Predicting emotion for audio file: {audio_file}")
    
    # Load model pipeline
    logger.info(f"Loading model pipeline from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except (FileNotFoundError, pickle.PickleError) as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please train a model first with --develop-models or --train")
        return None
    
    # Load audio file
    logger.info(f"Loading audio file {audio_file}")
    try:
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None
    
    # Preprocess audio
    logger.info("Preprocessing audio...")
    from src.preprocessing import AudioPreprocessor
    preprocessor = AudioPreprocessor(config_path=config_path)
    y_processed = preprocessor.preprocess_audio(y, sr)
    
    # Extract features
    logger.info("Extracting features...")
    from src.features import FeatureExtractor
    feature_extractor = FeatureExtractor(config_path=config_path)
    feature_vector = feature_extractor.extract_features_from_segment(y_processed, sr)
    
    # Reshape feature vector to 2D if needed
    if len(feature_vector.shape) == 1:
        feature_vector = feature_vector.reshape(1, -1)
    elif len(feature_vector.shape) > 2:
        feature_vector = feature_vector.reshape(1, -1)
    
    logger.info(f"Original feature vector shape: {feature_vector.shape}")
        
    # First apply scaling (StandardScaler)
    scaled_features = pipeline['scaler'].transform(feature_vector)
    logger.info(f"Scaled feature vector shape: {scaled_features.shape}")
    
    # Then apply feature selection if present
    if 'selected_features' in pipeline and pipeline['selected_features'] is not None:
        try:
            # Select features after scaling
            selected_indices = pipeline['selected_features']
            selected_features = scaled_features[:, selected_indices]
            logger.info(f"Feature selection applied: {len(selected_indices)} features selected")
            logger.info(f"Selected features shape: {selected_features.shape}")
            # Use selected features for prediction
            X = selected_features
        except Exception as e:
            logger.error(f"Error applying feature selection: {e}")
            raise
    else:
        # No feature selection - use all scaled features
        logger.info("No feature selection applied")
        X = scaled_features
    
    # Predict
    model = pipeline['model']
    y_pred = model.predict(X)
    
    # Convert prediction to emotion label
    emotion = pipeline['label_encoder'].inverse_transform(y_pred)[0]
    
    logger.info(f"Predicted emotion: {emotion}")
    
    # Get prediction probabilities if available
    try:
        proba = model.predict_proba(X)[0]
        classes = pipeline['label_encoder'].classes_
        
        # Print probabilities for each class
        logger.info("Prediction probabilities:")
        for emotion_class, prob in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
            logger.info(f"{emotion_class}: {prob:.4f}")
    except:
        logger.warning("Prediction probabilities not available")
    
    return emotion

def batch_predict(audio_dir=None, model_path=None, output_file=None):
    """
    Predict emotions for a batch of audio files
    
    Args:
        audio_dir (str): Directory containing audio files
        model_path (str): Path to model file
        output_file (str): Path to output file
    """
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if audio_dir is None:
        audio_dir = config['paths'].get('audio_dir', 'data/audio')
    
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], 'prediction_pipeline.pkl')
    
    if output_file is None:
        output_file = os.path.join(config['paths']['results'], 'batch_predictions.csv')
    
    logger.info(f"Batch predicting emotions for audio files in {audio_dir}")
    
    # Check if directory exists
    if not os.path.exists(audio_dir):
        logger.error(f"Directory not found: {audio_dir}")
        return
    
    # Load model pipeline
    logger.info(f"Loading model pipeline from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}")
        return
    
    # Initialize components
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Get audio files
    audio_files = []
    for ext in ['wav', 'mp3', 'ogg', 'flac']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, f'*.{ext}')))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize results
    results = []
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=None)
            
            # Preprocess audio
            y_processed = preprocessor.preprocess_audio(y, sr)
            
            # Extract features
            feature_vector = feature_extractor.extract_features_from_segment(y_processed, sr)
            
            # Reshape feature vector to 2D if needed
            if len(feature_vector.shape) == 1:
                feature_vector = feature_vector.reshape(1, -1)
            elif len(feature_vector.shape) > 2:
                feature_vector = feature_vector.reshape(1, -1)
            
            # First apply scaling (StandardScaler)
            scaled_features = pipeline['scaler'].transform(feature_vector)
            
            # Then apply feature selection if present
            if 'selected_features' in pipeline and pipeline['selected_features'] is not None:
                try:
                    # Select features after scaling
                    selected_indices = pipeline['selected_features']
                    selected_features = scaled_features[:, selected_indices]
                    # Use selected features for prediction
                    X = selected_features
                except Exception as e:
                    logger.error(f"Error applying feature selection: {e}")
                    raise
            else:
                # No feature selection - use all scaled features
                X = scaled_features
            
            # Predict
            y_pred = pipeline['model'].predict(X)
            emotion = pipeline['label_encoder'].inverse_transform(y_pred)[0]
            
            # Get probabilities if available
            confidence = 0.0
            if hasattr(pipeline['model'], 'predict_proba'):
                probas = pipeline['model'].predict_proba(X)[0]
                confidence = probas[y_pred[0]]
            
            # Add to results
            filename = os.path.basename(audio_file)
            results.append({
                'filename': filename,
                'emotion': emotion,
                'confidence': confidence
            })
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            # Add error to results
            filename = os.path.basename(audio_file)
            results.append({
                'filename': filename,
                'emotion': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved batch prediction results to {output_file}")
    
    # Print summary
    emotion_counts = results_df['emotion'].value_counts()
    logger.info("Emotion distribution in predictions:")
    for emotion, count in emotion_counts.items():
        logger.info(f"{emotion}: {count} ({count/len(results_df)*100:.1f}%)")
    
    return results_df

def check_gpu_capabilities():
    """
    Check GPU capabilities for machine learning
    """
    logger.info("Checking GPU capabilities for machine learning...")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        logger.warning("PyTorch not installed")
    
    try:
        import cupy
        logger.info(f"CuPy installed (needed for XGBoost GPU acceleration)")
        logger.info(f"CUDA version detected by CuPy: {cupy.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        logger.warning("CuPy not installed - XGBoost will use CPU")
    
    try:
        from cuml.ensemble import RandomForestClassifier
        logger.info("RAPIDS cuML installed (needed for RandomForest GPU acceleration)")
    except ImportError:
        logger.warning("RAPIDS cuML not installed - RandomForest will use CPU")
        logger.warning("Note: RAPIDS cuML is not officially supported on Windows")
        logger.warning("Consider using WSL2 or Docker for GPU acceleration with RAPIDS")
    
    try:
        import xgboost as xgb
        logger.info(f"XGBoost version: {xgb.__version__}")
        
        # Check if XGBoost was built with GPU support
        build_info = xgb.build_info()
        if 'USE_CUDA' in build_info and build_info['USE_CUDA'] == '1':
            logger.info("XGBoost was built with CUDA support")
        else:
            logger.warning("XGBoost was NOT built with CUDA support")
            
        logger.info(f"XGBoost build info: {build_info}")
    except (ImportError, Exception) as e:
        logger.warning(f"Error checking XGBoost: {e}")
    
    logger.info("\nFor XGBoost GPU acceleration on Windows:")
    logger.info("1. Install CUDA toolkit from NVIDIA website")
    logger.info("2. Install CuPy: pip install cupy-cuda11x (replace 11x with your CUDA version)")
    logger.info("\nFor RandomForest GPU acceleration (RAPIDS):")
    logger.info("This requires Linux or WSL2. For Windows, install WSL2 and then:")
    logger.info("1. conda create -n rapids-23.10 -c rapidsai -c conda-forge -c nvidia rapids=23.10 python=3.10 cuda-version=11.8")
    logger.info("2. Run your code inside the WSL2 environment")

def analyze_predictions(predictions_file=None, output_dir=None):
    """
    Analyze prediction results and create a confusion matrix
    
    Args:
        predictions_file (str): Path to predictions CSV file
        output_dir (str): Directory to save analysis results
    """
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if predictions_file is None:
        predictions_file = os.path.join(config['paths']['results'], 'batch_predictions.csv')
    
    if output_dir is None:
        output_dir = os.path.join(config['paths']['results'], 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing prediction results from {predictions_file}")
    
    # Load predictions
    try:
        predictions_df = pd.read_csv(predictions_file)
    except Exception as e:
        logger.error(f"Error loading predictions file: {e}")
        return
    
    logger.info(f"Loaded {len(predictions_df)} predictions")
    
    # Extract true emotions from filenames
    # CREMA-D files are named like: 1001_IEO_HAP_XX.wav where HAP is happy
    emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy', 
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    # Extract true emotions from filenames
    true_emotions = []
    for filename in predictions_df['filename']:
        parts = filename.split('_')
        if len(parts) >= 3:
            # Extract the emotion code
            emotion_code = parts[2]
            # Map to full emotion name
            true_emotion = emotion_map.get(emotion_code, 'unknown')
            true_emotions.append(true_emotion)
        else:
            # Handle filenames that don't match expected format
            true_emotions.append('unknown')
    
    # Add true emotions to dataframe
    predictions_df['true_emotion'] = true_emotions
    
    # Create a confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Filter out unknowns or rows with errors
    valid_df = predictions_df[predictions_df['true_emotion'] != 'unknown']
    
    # Generate confusion matrix
    cm = confusion_matrix(
        valid_df['true_emotion'], 
        valid_df['emotion'],
        labels=list(emotion_map.values())
    )
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=list(emotion_map.values()),
               yticklabels=list(emotion_map.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    
    # Save the confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Create and save the classification report
    report = classification_report(
        valid_df['true_emotion'], 
        valid_df['emotion'],
        labels=list(emotion_map.values()),
        output_dict=True
    )
    
    # Convert to DataFrame for easier viewing
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    logger.info(f"Saved classification report to {report_path}")
    
    # Calculate accuracy per emotion
    accuracy_per_emotion = {}
    for emotion in emotion_map.values():
        emotion_df = valid_df[valid_df['true_emotion'] == emotion]
        if len(emotion_df) > 0:
            accuracy = (emotion_df['emotion'] == emotion_df['true_emotion']).mean()
            accuracy_per_emotion[emotion] = accuracy
    
    # Display overall accuracy
    overall_accuracy = (valid_df['emotion'] == valid_df['true_emotion']).mean()
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # Display accuracy per emotion
    logger.info("Accuracy per emotion:")
    for emotion, accuracy in accuracy_per_emotion.items():
        logger.info(f"{emotion}: {accuracy:.4f}")
    
    # Find most common misclassifications
    misclassified = valid_df[valid_df['emotion'] != valid_df['true_emotion']]
    misclass_counts = misclassified.groupby(['true_emotion', 'emotion']).size().reset_index(name='count')
    misclass_counts = misclass_counts.sort_values('count', ascending=False)
    
    logger.info("Top misclassifications:")
    for _, row in misclass_counts.head(10).iterrows():
        logger.info(f"{row['true_emotion']} → {row['emotion']}: {row['count']} instances")
    
    return valid_df

def analyze_feature_importance(model_path=None, output_dir=None, top_n=20):
    """
    Analyze feature importance from the trained model
    
    Args:
        model_path (str): Path to the trained model pipeline
        output_dir (str): Directory to save feature importance visualizations
        top_n (int): Number of top features to visualize
    """
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], 'prediction_pipeline.pkl')
    
    if output_dir is None:
        output_dir = os.path.join(config['paths']['results'], 'feature_importance')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing feature importance from model: {model_path}")
    
    # Load model pipeline
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}")
        return
    
    # Extract model and feature names
    model = pipeline['model']
    
    # Create feature extractor to get feature names
    feature_extractor = FeatureExtractor()
    feature_names = feature_extractor.get_feature_names()
    
    # Check if we need to adjust feature names based on selected features
    if 'selected_features' in pipeline and pipeline['selected_features'] is not None:
        selected_indices = pipeline['selected_features']
        selected_feature_names = [feature_names[i] for i in selected_indices]
    else:
        selected_feature_names = feature_names
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models like RandomForest, XGBoost
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': importances
        })
    elif hasattr(model, 'coef_'):
        # For linear models like SVM, Logistic Regression
        importances = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': importances
        })
    else:
        logger.warning(f"Model type {type(model)} doesn't support feature importance extraction")
        return
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Features by Importance')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    logger.info(f"Saved feature importance plot to {plot_path}")
    
    # Save the full feature importance data
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(csv_path, index=False)
    logger.info(f"Saved feature importance data to {csv_path}")
    
    # Group features by type and analyze
    feature_types = {}
    for feature in feature_importance['feature']:
        if '_mfcc' in feature:
            feature_type = 'MFCC'
        elif '_chroma' in feature:
            feature_type = 'Chroma'
        elif '_contrast' in feature:
            feature_type = 'Spectral Contrast'
        elif '_centroid' in feature:
            feature_type = 'Spectral Centroid'
        elif '_bandwidth' in feature:
            feature_type = 'Spectral Bandwidth'
        elif '_rolloff' in feature:
            feature_type = 'Spectral Rolloff'
        elif '_zcr' in feature:
            feature_type = 'Zero Crossing Rate'
        elif '_energy' in feature:
            feature_type = 'Energy'
        elif '_rmse' in feature:
            feature_type = 'RMSE'
        else:
            feature_type = 'Other'
        
        if feature_type not in feature_types:
            feature_types[feature_type] = 0
        feature_types[feature_type] += feature_importance.loc[feature_importance['feature'] == feature, 'importance'].iloc[0]
    
    # Convert to DataFrame for visualization
    feature_type_importance = pd.DataFrame({
        'feature_type': list(feature_types.keys()),
        'importance': list(feature_types.values())
    })
    feature_type_importance = feature_type_importance.sort_values('importance', ascending=False)
    
    # Plot feature type importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature_type', data=feature_type_importance)
    plt.title('Feature Type Importance')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'feature_type_importance.png')
    plt.savefig(plot_path)
    logger.info(f"Saved feature type importance plot to {plot_path}")
    
    # Return feature importance data
    return feature_importance

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Audio Emotion Recognition System')
    
    parser.add_argument('--check-env', action='store_true', help='Check environment and dependencies')
    parser.add_argument('--verify-structure', action='store_true', help='Verify project structure')
    parser.add_argument('--process-dataset', action='store_true', help='Process audio dataset')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--develop-models', action='store_true', help='Develop and evaluate models')
    parser.add_argument('--cross-validate', action='store_true', help='Cross-validate models')
    parser.add_argument('--predict', action='store_true', help='Predict emotion for audio file')
    parser.add_argument('--batch-predict', action='store_true', help='Batch predict emotions for audio files')
    parser.add_argument('--check-gpu', action='store_true', help='Check GPU capabilities')
    parser.add_argument('--analyze-predictions', action='store_true', help='Analyze prediction results')
    parser.add_argument('--analyze-feature-importance', action='store_true', help='Analyze feature importance')
    parser.add_argument('--statistical-analysis', action='store_true', help='Perform statistical analysis on model results')
    
    parser.add_argument('--audio-file', type=str, help='Audio file for prediction')
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--feature-path', type=str, help='Path to feature file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--predictions-file', type=str, help='Predictions file')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--no-feature-selection', action='store_true', help='Disable feature selection')
    
    args = parser.parse_args()
    
    # Check environment
    if args.check_env:
        check_environment()
    
    # Verify project structure
    if args.verify_structure:
        verify_project_structure()
    
    # Process dataset
    if args.process_dataset:
        process_dataset(limit=args.limit)
    
    # Train models
    if args.train:
        train_models(feature_path=args.feature_path, model_output=args.output_dir, use_gpu=args.use_gpu)
    
    # Evaluate models
    if args.evaluate:
        evaluate_model(model_path=args.model_path, feature_path=args.feature_path)
    
    # Develop and evaluate models
    if args.develop_models:
        develop_models(
            feature_path=args.feature_path,
            model_output_dir=args.output_dir,
            use_feature_selection=not args.no_feature_selection
        )
    
    # Cross-validate models
    if args.cross_validate:
        cross_validate_models(
            feature_path=args.feature_path,
            n_folds=args.n_folds,
            output_dir=args.output_dir,
            use_feature_selection=not args.no_feature_selection
        )
    
    # Predict emotion
    if args.predict:
        if not args.audio_file:
            print("Please specify an audio file with --audio-file")
            return
        
        emotion = predict_emotion(
            audio_file=args.audio_file,
            model_path=args.model_path
        )
        
        if emotion:
            print(f"Predicted emotion: {emotion}")
    
    # Batch predict emotions
    if args.batch_predict:
        if not args.audio_dir:
            print("Please specify an audio directory with --audio-dir")
            return
        
        batch_predict(
            audio_dir=args.audio_dir,
            model_path=args.model_path,
            output_file=args.predictions_file
        )
    
    # Check GPU capabilities
    if args.check_gpu:
        check_gpu_capabilities()
    
    # Analyze predictions
    if args.analyze_predictions:
        analyze_predictions(
            predictions_file=args.predictions_file,
            output_dir=args.output_dir
        )
    
    # Analyze feature importance
    if args.analyze_feature_importance:
        analyze_feature_importance(
            model_path=args.model_path,
            output_dir=args.output_dir
        )
    
    # Perform statistical analysis
    if args.statistical_analysis:
        from src.statistical_analysis import run_statistical_analysis
        run_statistical_analysis(
            cv_results_path=os.path.join(args.output_dir, 'cv_results.pkl') if args.output_dir else None,
            output_dir=os.path.join(args.output_dir, 'statistical_analysis') if args.output_dir else None
        )

if __name__ == "__main__":
    main()