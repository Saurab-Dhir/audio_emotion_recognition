import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import load_config
from src.features import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    """
    Emotion recognition model trainer
    """
    def __init__(self, config_path: str = "config.yaml", use_gpu: bool = False):
        """
        Initialize model trainer
        
        Args:
            config_path (str): Path to configuration file
            use_gpu (bool): Whether to use GPU acceleration when available
        """
        self.config = load_config(config_path)
        self.model_config = self.config['model']
        self.feature_extractor = FeatureExtractor(config_path)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0.0
        
        # Initialize GPU flags
        self.use_gpu = use_gpu
        self.cuml_available = False
        self.xgb_gpu_available = False
        
        # Check for actual GPU capabilities
        if self.use_gpu:
            # Check for XGBoost GPU support
            try:
                import cupy
                logger.info(f"CuPy is available - XGBoost GPU acceleration possible")
                self.xgb_gpu_available = True
            except ImportError:
                logger.warning("CuPy not found - XGBoost will use CPU")
                self.xgb_gpu_available = False
                
            # Check for Random Forest GPU support
            try:
                from cuml.ensemble import RandomForestClassifier
                logger.info("RAPIDS cuML is available - Random Forest GPU acceleration possible")
                self.cuml_available = True
            except ImportError:
                logger.warning("RAPIDS cuML not found - Random Forest will use CPU")
                self.cuml_available = False
                
            # Log overall GPU acceleration status
            if not (self.xgb_gpu_available or self.cuml_available):
                logger.warning("GPU acceleration requested but no GPU libraries are available - using CPU only")
                self.use_gpu = False
            else:
                logger.info(f"GPU acceleration enabled: XGBoost={self.xgb_gpu_available}, RandomForest={self.cuml_available}")
        else:
            logger.info("GPU acceleration not requested - using CPU only")
        
    def load_feature_dataset(self, feature_path: str) -> pd.DataFrame:
        """
        Load feature dataset
        
        Args:
            feature_path (str): Path to feature dataset
            
        Returns:
            pd.DataFrame: Feature dataset
        """
        try:
            feature_df = self.feature_extractor.load_features(feature_path)
            logger.info(f"Loaded feature dataset with shape {feature_df.shape}")
            return feature_df
        except Exception as e:
            logger.error(f"Error loading feature dataset: {e}")
            raise
    
    def prepare_data(self, feature_df: pd.DataFrame, target_col: str = 'emotion', 
                  test_size: Optional[float] = None, random_state: Optional[int] = None) -> Tuple:
        """
        Prepare data for model training
        
        Args:
            feature_df (pd.DataFrame): Feature dataset
            target_col (str): Target column name
            test_size (Optional[float]): Test set size
            random_state (Optional[int]): Random state
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test, feature_names
        """
        if test_size is None:
            test_size = self.model_config.get('test_size', 0.2)
        
        if random_state is None:
            random_state = self.model_config.get('random_state', 42)
        
        # Extract features and target
        X = np.vstack(feature_df['feature_vector'].values)
        y = feature_df[target_col].values
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Store class labels
        self.classes_ = self.label_encoder.classes_
        
        logger.info(f"Prepared data with {X_train.shape[1]} features")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Classes: {self.classes_}")
        
        return X_train, X_test, y_train, y_test, list(range(X_train.shape[1]))
    
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, method: str = 'selectk', 
                      k: int = 100) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Select features
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features
            method (str): Feature selection method
            k (int): Number of features to select
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[int]]: X_train_selected, X_test_selected, selected_features
        """
        if method == 'selectk':
            # Select k best features using ANOVA F-value
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature indices
            selected_features = np.where(selector.get_support())[0].tolist()
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k, step=0.1)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature indices
            selected_features = np.where(selector.get_support())[0].tolist()
            
        else:
            # No feature selection
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = list(range(X_train.shape[1]))
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return X_train_selected, X_test_selected, selected_features
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                tune_hyperparams: bool = True) -> Tuple[SVC, Dict[str, float]]:
        """
        Train SVM model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test/validation features
            y_test (np.ndarray): Test/validation target
            tune_hyperparams (bool): Whether to tune hyperparameters
            
        Returns:
            Tuple[SVC, Dict[str, float]]: Trained SVM model and evaluation metrics
        """
        logger.info("Training SVM model")
        model_name = "svm"
        
        try:
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear', 'poly']
                }
                
                # Create model
                base_model = SVC(probability=True, random_state=self.model_config.get('random_state', 42))
                
                # Create grid search
                cv_folds = self.model_config.get('hyperparameter_tuning', {}).get('cv_folds', 5)
                model = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                
                # Train model
                model.fit(X_train, y_train)
                
                logger.info(f"SVM best parameters: {model.best_params_}")
                
            else:
                # Create model with default parameters
                model = SVC(probability=True, random_state=self.model_config.get('random_state', 42))
                
                # Train model
                model.fit(X_train, y_train)
            
            # Evaluate model
            model_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
        except Exception as e:
            logger.error(f"Error training SVM model: {e}")
            # Return empty model and metrics in case of failure
            model = None
            model_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        
        return model, model_metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, tune_hyperparams=True):
        """
        Train Random Forest model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test/validation features
            y_test (np.ndarray): Test/validation target
            tune_hyperparams (bool): Whether to tune hyperparameters
        """
        logger.info("Training Random Forest model")
        model_name = "random_forest"
        use_gpu = False
        
        try:
            # Try to use GPU-accelerated Random Forest if RAPIDS is available
            if self.use_gpu and self.cuml_available:
                try:
                    from cuml.ensemble import RandomForestClassifier as cuRF
                    logger.info("Training Random Forest using GPU (RAPIDS cuML)")
                    rf_model = cuRF(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42
                    )
                    use_gpu = True
                except (ImportError, Exception) as e:
                    logger.warning(f"Error initializing GPU Random Forest: {e}. Using CPU RandomForest.")
                    rf_model = RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42,
                        n_jobs=-1
                    )
            else:
                if self.use_gpu and not self.cuml_available:
                    logger.info("GPU acceleration requested for Random Forest but RAPIDS cuML not available. Using CPU.")
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Add hyperparameter tuning if requested
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Create grid search
                cv_folds = self.model_config.get('hyperparameter_tuning', {}).get('cv_folds', 5)
                rf_model = GridSearchCV(rf_model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
            
            # Train model
            logger.info(f"Fitting Random Forest model (GPU: {use_gpu})")
            rf_model.fit(X_train, y_train)
            
            # If model is a GridSearchCV instance, log best parameters and extract best estimator
            if hasattr(rf_model, 'best_params_'):
                logger.info(f"Random Forest best parameters: {rf_model.best_params_}")
                rf_model = rf_model.best_estimator_
            
            # Evaluate model
            model_metrics = self.evaluate_model(rf_model, X_test, y_test, model_name, use_gpu)
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            # Return empty model and metrics in case of failure
            rf_model = None
            model_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        
        return rf_model, model_metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, tune_hyperparams=True, handle_imbalance=True):
        """
        Train XGBoost model with optional GPU support
        """
        logger.info("Training XGBoost model")
        model_name = "xgboost"
        use_gpu = False
        
        try:
            # Calculate class weights if handling imbalance
            if handle_imbalance:
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                sample_weights = np.ones(y_train.shape[0], dtype=np.float32)
                for i, val in enumerate(np.unique(y_train)):
                    sample_weights[y_train == val] = class_weights[i]
            else:
                sample_weights = None
            
            # Set GPU parameters if GPU is available and enabled
            gpu_params = {}
            if self.use_gpu and self.xgb_gpu_available:
                try:
                    import cupy  # Verify cupy is available
                    logger.info("Training XGBoost using GPU")
                    gpu_params = {
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    }
                    use_gpu = True
                except (ImportError, Exception) as e:
                    logger.warning(f"Error initializing XGBoost GPU parameters: {e}. Using CPU XGBoost.")
            elif self.use_gpu and not self.xgb_gpu_available:
                logger.info("GPU acceleration requested for XGBoost but CuPy not available. Using CPU.")
            
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
                
                # Create model with GPU params if available
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    **gpu_params
                )
                
                # Create grid search
                cv_folds = 5
                model = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                
                # Train model
                logger.info(f"Fitting XGBoost model with hyperparameter tuning (GPU: {use_gpu})")
                model.fit(X_train, y_train, sample_weight=sample_weights)
                
                logger.info(f"XGBoost best parameters: {model.best_params_}")
                model = model.best_estimator_
                
            else:
                # Create model with default parameters
                model = xgb.XGBClassifier(
                    n_estimators=100, 
                    random_state=42,
                    **gpu_params
                )
                
                # Train model
                logger.info(f"Fitting XGBoost model (GPU: {use_gpu})")
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Evaluate model
            model_metrics = self.evaluate_model(model, X_test, y_test, model_name, use_gpu)
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            # Return empty model and metrics in case of failure
            model = None
            model_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        
        return model, model_metrics
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple models
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        models = {}
        
        # Get list of models to train
        models_to_train = self.model_config.get('models_to_train', ['svm', 'random_forest', 'xgboost'])
        
        # Check if hyperparameter tuning is enabled
        tune_hyperparams = self.model_config.get('hyperparameter_tuning', {}).get('perform', False)
        
        # Split training data to create a validation set
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        # Train models
        for model_name in models_to_train:
            logger.info(f"Training {model_name} model...")
            
            if model_name == 'svm':
                models[model_name] = self.train_svm(X_train_split, y_train_split, X_val, y_val, tune_hyperparams)
            elif model_name == 'random_forest':
                models[model_name] = self.train_random_forest(X_train_split, y_train_split, X_val, y_val, tune_hyperparams)
            elif model_name == 'xgboost':
                models[model_name] = self.train_xgboost(X_train_split, y_train_split, X_val, y_val, tune_hyperparams, True)
            else:
                logger.warning(f"Unknown model: {model_name}")
        
        self.models = models
        return models
    
    def evaluate_model(self, model, X_test, y_test, model_name, use_gpu=False):
        """
        Evaluate model with multiple metrics
        """
        if use_gpu and hasattr(model, 'device') and model.device == 'cuda':
            logger.info(f"Evaluating {model_name} using GPU")
            # For XGBoost with device='cuda', ensure data is on same device
            try:
                import cupy as cp
                if not isinstance(X_test, cp.ndarray):
                    logger.info(f"Converting test data to GPU for {model_name}")
                    # Only convert to GPU if model is XGBoost with cuda device
                    if isinstance(model, xgb.XGBModel):
                        # Use DMatrix for GPU prediction
                        dtest = xgb.DMatrix(X_test, label=y_test)
                        y_pred = model.predict(dtest)
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            y_pred = y_pred.argmax(axis=1)
                    else:
                        y_pred = model.predict(X_test)
            except (ImportError, Exception) as e:
                logger.warning(f"Error using GPU for prediction: {e}. Using standard prediction.")
                y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for all models
        """
        metrics = {}
        
        for model_name, model_tuple in self.models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            # Get the model from the tuple (model, metrics)
            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
            
            # Evaluate model
            model_metrics = self.evaluate_model(model, X_test, y_test, model_name, True)
            metrics[model_name] = model_metrics
            
            # Log metrics
            logger.info(f"{model_name} metrics: {model_metrics}")
            
            # Update best model
            if model_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = model_metrics['accuracy']
                self.best_model = model
                self.best_model_name = model_name
        
        logger.info(f"Best model: {self.best_model_name} with accuracy {self.best_accuracy:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, 
                           model_name: str = None, save_path: str = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Model name
            save_path (str): Path to save the plot
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model_tuple = self.models.get(model_name)
        if model_tuple is None:
            model = self.best_model
        else:
            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
        
        if model is None:
            logger.warning("No model available for confusion matrix")
            return
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(f'Confusion Matrix - {model_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str = None, save_path: str = None) -> None:
        """
        Save model
        
        Args:
            model_name (str): Model name
            save_path (str): Path to save the model
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model_tuple = self.models.get(model_name)
        if model_tuple is None:
            model = self.best_model
        else:
            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
        
        if model is None:
            logger.warning("No model available to save")
            return
        
        if save_path is None:
            save_path = os.path.join(self.config['paths']['models'], f"{model_name}_model.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'classes': self.classes_
            }, f)
        
        logger.info(f"Saved model to {save_path}")
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load model
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            Dict[str, Any]: Loaded model and associated objects
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components
        model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.classes_ = model_data['classes']
        
        logger.info(f"Loaded model from {model_path}")
        
        return model_data
    
    def predict(self, features: np.ndarray, model: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict emotion from features
        
        Args:
            features (np.ndarray): Features
            model (Any): Model to use for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and probabilities
        """
        if model is None:
            model = self.best_model
        
        # Handle case where model is tuple (model, metrics)
        if isinstance(model, tuple) and len(model) > 0:
            model = model[0]
        
        if model is None:
            logger.warning("No model available for prediction")
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        y_pred = model.predict(features_scaled)
        
        # Get prediction probabilities
        try:
            y_proba = model.predict_proba(features_scaled)
        except:
            # Some models might not have predict_proba method
            y_proba = None
        
        # Convert predicted class indices back to labels
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred_labels, y_proba
    
    def predict_from_audio(self, audio_data: Tuple[np.ndarray, int], model: Any = None) -> str:
        """
        Predict emotion from audio data
        
        Args:
            audio_data (Tuple[np.ndarray, int]): Audio data and sample rate
            model (Any): Model to use for prediction
            
        Returns:
            str: Predicted emotion
        """
        from preprocessing import AudioPreprocessor
        
        # Handle case where model is tuple (model, metrics)
        if isinstance(model, tuple) and len(model) > 0:
            model = model[0]
            
        y, sr = audio_data
        
        # Preprocess audio
        preprocessor = AudioPreprocessor()
        y_preprocessed = preprocessor.preprocess_audio(y, sr)
        
        # Extract features
        features = self.feature_extractor.extract_features_from_segment(y_preprocessed, sr)
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Predict
        y_pred_labels, _ = self.predict(features, model)
        
        if y_pred_labels is None or len(y_pred_labels) == 0:
            return "Unknown"
        
        return y_pred_labels[0]
    
    def run_pipeline(self, feature_path: str, output_dir: str = None, feature_selection: str = None,
                  k_features: int = 100) -> Dict[str, Any]:
        """
        Run the complete model training pipeline
        
        Args:
            feature_path (str): Path to feature dataset
            output_dir (str): Output directory for results
            feature_selection (str): Feature selection method
            k_features (int): Number of features to select
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        # Load feature dataset
        feature_df = self.load_feature_dataset(feature_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(feature_df)
        
        # Feature selection
        if feature_selection:
            X_train, X_test, selected_features = self.select_features(
                X_train, y_train, X_test, method=feature_selection, k=k_features
            )
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        metrics = self.evaluate_models(X_test, y_test)
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config['paths']['results'], 'model_evaluation')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        for model_name in self.models.keys():
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            self.plot_confusion_matrix(X_test, y_test, model_name, cm_path)
        
        # Save models
        for model_name in self.models.keys():
            model_path = os.path.join(self.config['paths']['models'], f"{model_name}_model.pkl")
            self.save_model(model_name, model_path)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "model_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info(f"Pipeline completed. Results saved to {output_dir}")
        
        return {
            'metrics': metrics,
            'best_model_name': self.best_model_name,
            'best_accuracy': self.best_accuracy
        }

def main():
    """
    Main function for testing the model trainer
    """
    # Initialize model trainer
    model_trainer = EmotionModelTrainer()
    
    # Set paths
    feature_path = os.path.join('data', 'features', 'cremad_features.pkl')
    output_dir = os.path.join('results', 'model_evaluation')
    
    # Run pipeline with a small subset for testing
    try:
        results = model_trainer.run_pipeline(
            feature_path=feature_path,
            output_dir=output_dir,
            feature_selection='selectk',
            k_features=50
        )
        
        # Print results
        logger.info("Model training results:")
        logger.info(f"Best model: {results['best_model_name']}")
        logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
        
        for model_name, metrics in results['metrics'].items():
            logger.info(f"{model_name} metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()