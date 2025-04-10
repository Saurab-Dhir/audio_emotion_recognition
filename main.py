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
    
    # Train SVM model
    logger.info("Training SVM model...")
    svm_model = trainer.train_svm(X_train, y_train, tune_hyperparams=tune_hyperparams)
    models['svm'] = svm_model
    
    # Train Random Forest model
    logger.info("Training Random Forest model...")
    rf_model = trainer.train_random_forest(X_train, y_train, tune_hyperparams=tune_hyperparams)
    models['random_forest'] = rf_model
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_model = trainer.train_xgboost(X_train, y_train, tune_hyperparams=tune_hyperparams)
    models['xgboost'] = xgb_model
    
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

def main():
    """
    Main function to run the emotion recognition system
    """
    parser = argparse.ArgumentParser(description="Audio-Based Emotion Recognition System")
    parser.add_argument("--check-env", action="store_true", help="Check environment setup")
    parser.add_argument("--test-loader", action="store_true", help="Test CREMA-D data loader")
    parser.add_argument("--verify-structure", action="store_true", help="Verify project structure")
    parser.add_argument("--process", action="store_true", help="Process dataset through entire pipeline")
    parser.add_argument("--train", action="store_true", help="Train models on processed features")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--develop-models", action="store_true", help="Develop and evaluate ML models")
    parser.add_argument("--limit", type=int, help="Limit number of samples to process")
    parser.add_argument("--output", type=str, help="Output path for processed features")
    parser.add_argument("--feature-path", type=str, help="Path to feature dataset for training/evaluation")
    parser.add_argument("--model-path", type=str, help="Path to save/load model")
    parser.add_argument("--model-dir", type=str, help="Directory to save multiple models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (default: use GPU if available)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--no-feature-selection", action="store_true", help="Disable feature selection")
    parser.add_argument("--no-hyperparameter-tuning", action="store_true", help="Disable hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories(config)
    
    if args.check_env:
        check_environment()
        
    if args.test_loader:
        test_cremad_loader(config, limit=args.limit or 5)
        
    if args.verify_structure:
        verify_project_structure()
        
    if args.process:
        process_dataset(limit=args.limit, output_path=args.output)
        
    if args.train:
        train_models(
            feature_path=args.feature_path,
            model_output=args.model_path,
            use_gpu=not args.cpu,
            num_epochs=args.epochs
        )
        
    if args.evaluate:
        evaluate_model(
            model_path=args.model_path,
            feature_path=args.feature_path
        )
        
    if args.develop_models:
        develop_models(
            feature_path=args.feature_path,
            model_output_dir=args.model_dir,
            use_feature_selection=not args.no_feature_selection,
            tune_hyperparams=not args.no_hyperparameter_tuning
        )
    
    # If no specific arguments, run basic verification
    if not any([args.check_env, args.test_loader, args.verify_structure, args.process, args.train, args.evaluate, args.develop_models]):
        check_environment()
        test_cremad_loader(config, limit=5)
        verify_project_structure()
    
    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main()