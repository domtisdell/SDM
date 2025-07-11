"""
Adaptive Ensemble Weighting System for Steel Demand Forecasting

This module provides intelligent ensemble combination strategies:
- Performance-based weighting using cross-validation scores
- Category-specific weight optimization
- Fallback hierarchies for failed models
- Dynamic weight adjustment based on prediction confidence
- Model diversity assessment and balancing

Designed to replace fixed 60/40 weighting in Track A system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.special import softmax

class AdaptiveEnsembleWeighting:
    """
    Intelligent ensemble weighting system based on model performance and diversity.
    
    Features:
    - Cross-validation based weight calculation
    - Category-specific optimization
    - Model confidence assessment
    - Diversity-performance balance
    - Fallback strategies for failed models
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize adaptive ensemble system.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Weight storage
        self.category_weights = {}
        self.global_weights = {}
        self.fallback_hierarchy = ['XGBoost', 'RandomForest', 'Ridge', 'Lasso', 'ElasticNet']
        
        # Performance tracking
        self.model_scores = {}
        self.diversity_metrics = {}
        self.ensemble_performance = {}
        
        # Weighting strategies
        self.weighting_strategies = {
            'performance_only': self._performance_based_weights,
            'performance_diversity': self._performance_diversity_weights,
            'bayesian_optimal': self._bayesian_optimal_weights,
            'robust_average': self._robust_average_weights
        }
        
        # Configuration
        self.min_weight_threshold = 0.05  # Minimum weight for any model
        self.max_weight_threshold = 0.70  # Maximum weight for any model
        self.diversity_weight = 0.2  # Weight given to diversity vs pure performance
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        return logging.getLogger(__name__)
    
    def calculate_cross_validation_scores(self, models: Dict[str, Any], X: pd.DataFrame, 
                                        y: pd.Series, category: str) -> Dict[str, float]:
        """
        Calculate cross-validation scores for all models.
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary of CV scores for each model
        """
        self.logger.info(f"Calculating CV scores for {category} ({len(models)} models)")
        
        cv_scores = {}
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        for model_name, model_result in models.items():
            try:
                model = model_result['model']
                
                # For regularized models, use preprocessed features
                if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                    # Assume preprocessing was done during training
                    X_processed = X  # Should be preprocessed already
                else:
                    # Tree models use raw features
                    X_processed = X
                
                # Cross-validation scoring
                cv_mae_scores = cross_val_score(
                    model, X_processed, y,
                    cv=cv_strategy,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1  # Use all available cores
                )
                
                cv_r2_scores = cross_val_score(
                    model, X_processed, y,
                    cv=cv_strategy,
                    scoring='r2',
                    n_jobs=-1  # Use all available cores
                )
                
                # Combined score: emphasize R² but penalize high MAE
                mean_mae = -np.mean(cv_mae_scores)
                mean_r2 = np.mean(cv_r2_scores)
                std_r2 = np.std(cv_r2_scores)  # Stability metric
                
                # Combined score: R² * (1 - normalized_MAE) - penalty for instability
                if y.std() > 0:
                    normalized_mae = mean_mae / y.std()
                else:
                    normalized_mae = 0
                
                combined_score = mean_r2 * (1 - min(normalized_mae, 1.0)) - 0.1 * std_r2
                
                cv_scores[model_name] = {
                    'mae': mean_mae,
                    'r2': mean_r2,
                    'r2_std': std_r2,
                    'combined_score': combined_score,
                    'stability': 1 / (1 + std_r2)  # Higher is better
                }
                
                self.logger.debug(f"{model_name} - MAE: {mean_mae:.3f}, R²: {mean_r2:.3f}±{std_r2:.3f}, "
                                f"Score: {combined_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"CV scoring failed for {model_name}: {str(e)}")
                cv_scores[model_name] = {
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'r2_std': float('inf'),
                    'combined_score': -float('inf'),
                    'stability': 0.0
                }
        
        # Store scores
        self.model_scores[category] = cv_scores
        return cv_scores
    
    def calculate_diversity_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate diversity metrics between model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(predictions) < 2:
            return {'mean_correlation': 0.0, 'diversity_score': 1.0}
        
        model_names = list(predictions.keys())
        correlations = []
        
        # Calculate pairwise correlations
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pred_i = predictions[model_names[i]]
                pred_j = predictions[model_names[j]]
                
                if len(pred_i) > 1 and len(pred_j) > 1:
                    corr = np.corrcoef(pred_i, pred_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            mean_correlation = np.mean(correlations)
            diversity_score = 1 - mean_correlation  # Higher diversity = lower correlation
        else:
            mean_correlation = 0.0
            diversity_score = 1.0
        
        return {
            'mean_correlation': mean_correlation,
            'diversity_score': diversity_score,
            'n_comparisons': len(correlations)
        }
    
    def _performance_based_weights(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate weights based purely on performance scores."""
        if not scores:
            return {}
        
        combined_scores = {name: score_dict['combined_score'] for name, score_dict in scores.items()}
        
        # Convert to positive values and apply softmax
        min_score = min(combined_scores.values())
        if min_score < 0:
            combined_scores = {name: score - min_score + 0.1 for name, score in combined_scores.items()}
        
        score_array = np.array(list(combined_scores.values()))
        weights = softmax(score_array * 2)  # Temperature = 0.5 for sharper distribution
        
        return dict(zip(combined_scores.keys(), weights))
    
    def _performance_diversity_weights(self, scores: Dict[str, Dict[str, float]], 
                                     predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate weights balancing performance and diversity."""
        if not scores:
            return {}
        
        # Performance weights
        perf_weights = self._performance_based_weights(scores)
        
        # Diversity adjustment
        diversity_metrics = self.calculate_diversity_metrics(predictions)
        diversity_bonus = diversity_metrics['diversity_score']
        
        # Combine performance and diversity
        adjusted_weights = {}
        for model_name, perf_weight in perf_weights.items():
            if model_name in predictions:
                # Boost weight for diverse models
                diversity_factor = 1 + self.diversity_weight * diversity_bonus
                adjusted_weights[model_name] = perf_weight * diversity_factor
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {name: weight / total_weight for name, weight in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _bayesian_optimal_weights(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate optimal weights using Bayesian optimization approach."""
        if not scores:
            return {}
        
        model_names = list(scores.keys())
        n_models = len(model_names)
        
        if n_models == 1:
            return {model_names[0]: 1.0}
        
        # Objective function: minimize weighted prediction error
        def objective(weights):
            # Ensure weights sum to 1 and are positive
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # Calculate weighted score (negative because we minimize)
            weighted_score = 0
            for i, model_name in enumerate(model_names):
                model_score = scores[model_name]['combined_score']
                weighted_score += weights[i] * model_score
            
            return -weighted_score  # Minimize negative score = maximize score
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: minimum and maximum weight constraints
        bounds = [(self.min_weight_threshold, self.max_weight_threshold) for _ in range(n_models)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
                return dict(zip(model_names, optimal_weights))
            else:
                self.logger.warning("Bayesian optimization failed, falling back to performance-based weights")
                return self._performance_based_weights(scores)
                
        except Exception as e:
            self.logger.warning(f"Bayesian optimization error: {str(e)}, using performance-based weights")
            return self._performance_based_weights(scores)
    
    def _robust_average_weights(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate robust weights using multiple strategies and averaging."""
        if not scores:
            return {}
        
        # Calculate weights using multiple strategies
        perf_weights = self._performance_based_weights(scores)
        bayesian_weights = self._bayesian_optimal_weights(scores)
        
        # Average the strategies
        all_models = set(perf_weights.keys()) | set(bayesian_weights.keys())
        robust_weights = {}
        
        for model in all_models:
            weights_list = []
            if model in perf_weights:
                weights_list.append(perf_weights[model])
            if model in bayesian_weights:
                weights_list.append(bayesian_weights[model])
            
            if weights_list:
                robust_weights[model] = np.mean(weights_list)
        
        # Renormalize
        total_weight = sum(robust_weights.values())
        if total_weight > 0:
            robust_weights = {name: weight / total_weight for name, weight in robust_weights.items()}
        
        return robust_weights
    
    def calculate_adaptive_weights(self, models: Dict[str, Any], X: pd.DataFrame, 
                                 y: pd.Series, category: str, 
                                 strategy: str = 'performance_diversity') -> Dict[str, float]:
        """
        Calculate adaptive ensemble weights for a specific category.
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target variable  
            category: Steel category name
            strategy: Weighting strategy to use
            
        Returns:
            Dictionary of model weights
        """
        self.logger.info(f"Calculating adaptive weights for {category} using {strategy} strategy")
        
        # Calculate cross-validation scores
        cv_scores = self.calculate_cross_validation_scores(models, X, y, category)
        
        # Filter out failed models
        valid_models = {name: score for name, score in cv_scores.items() 
                       if score['combined_score'] > -float('inf')}
        
        if not valid_models:
            self.logger.warning(f"No valid models for {category}, using fallback weights")
            return self._get_fallback_weights(models)
        
        # Get predictions for diversity calculation
        predictions = {}
        for model_name, model_result in models.items():
            if model_name in valid_models:
                try:
                    model = model_result['model']
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        predictions[model_name] = pred
                except Exception as e:
                    self.logger.warning(f"Could not get predictions from {model_name}: {str(e)}")
        
        # Calculate weights using specified strategy
        if strategy in self.weighting_strategies:
            if strategy == 'performance_diversity':
                weights = self._performance_diversity_weights(valid_models, predictions)
            else:
                weights = self.weighting_strategies[strategy](valid_models)
        else:
            self.logger.warning(f"Unknown strategy {strategy}, using performance_only")
            weights = self._performance_based_weights(valid_models)
        
        # Apply weight constraints
        weights = self._apply_weight_constraints(weights)
        
        # Store weights
        self.category_weights[category] = weights
        
        # Log results
        self.logger.info(f"Adaptive weights for {category}:")
        for model_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            score = valid_models.get(model_name, {}).get('combined_score', 0)
            self.logger.info(f"  {model_name}: {weight:.3f} (score: {score:.3f})")
        
        return weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        if not weights:
            return weights
        
        # Apply minimum weight threshold
        for model_name in weights:
            if weights[model_name] < self.min_weight_threshold:
                weights[model_name] = self.min_weight_threshold
        
        # Renormalize after minimum adjustment
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Apply maximum weight threshold
        constrained = False
        for model_name in weights:
            if weights[model_name] > self.max_weight_threshold:
                weights[model_name] = self.max_weight_threshold
                constrained = True
        
        # Renormalize after maximum adjustment
        if constrained:
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def _get_fallback_weights(self, models: Dict[str, Any]) -> Dict[str, float]:
        """Get fallback weights when optimization fails."""
        model_names = list(models.keys())
        
        # Use fallback hierarchy
        available_models = [model for model in self.fallback_hierarchy if model in model_names]
        
        if not available_models:
            # Equal weights if no hierarchy matches
            equal_weight = 1.0 / len(model_names)
            return {name: equal_weight for name in model_names}
        
        # Weighted by position in hierarchy (earlier = higher weight)
        weights = {}
        total_weight = sum(range(1, len(available_models) + 1))
        
        for i, model_name in enumerate(available_models):
            hierarchy_weight = (len(available_models) - i) / total_weight
            weights[model_name] = hierarchy_weight
        
        # Zero weight for models not in hierarchy
        for model_name in model_names:
            if model_name not in weights:
                weights[model_name] = 0.0
        
        return weights
    
    def create_ensemble_prediction(self, predictions: Dict[str, np.ndarray], 
                                 weights: Dict[str, float]) -> np.ndarray:
        """
        Create ensemble prediction using weighted combination.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            
        Returns:
            Ensemble prediction array
        """
        if not predictions or not weights:
            return np.array([])
        
        # Filter to models with both predictions and weights
        valid_models = set(predictions.keys()) & set(weights.keys())
        
        if not valid_models:
            self.logger.warning("No valid models for ensemble, using first available prediction")
            return next(iter(predictions.values()))
        
        # Calculate weighted ensemble
        ensemble_pred = None
        total_weight = 0
        
        for model_name in valid_models:
            weight = weights[model_name]
            pred = predictions[model_name]
            
            if weight > 0:
                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred
                total_weight += weight
        
        if ensemble_pred is not None and total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
            return np.maximum(ensemble_pred, 0)  # Ensure non-negative
        else:
            # Fallback to simple average
            valid_predictions = [predictions[model] for model in valid_models]
            return np.maximum(np.mean(valid_predictions, axis=0), 0)
    
    def evaluate_ensemble_performance(self, y_true: pd.Series, 
                                    predictions: Dict[str, np.ndarray],
                                    weights: Dict[str, float],
                                    category: str) -> Dict[str, float]:
        """
        Evaluate ensemble performance against individual models.
        
        Args:
            y_true: True values
            predictions: Model predictions
            weights: Model weights
            category: Steel category name
            
        Returns:
            Dictionary with ensemble and individual model metrics
        """
        # Ensemble prediction
        ensemble_pred = self.create_ensemble_prediction(predictions, weights)
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'mae': mean_absolute_error(y_true, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, ensemble_pred)),
            'r2': r2_score(y_true, ensemble_pred),
            'mape': np.mean(np.abs((y_true - ensemble_pred) / (y_true + 1e-6))) * 100
        }
        
        # Individual model metrics
        individual_metrics = {}
        for model_name, pred in predictions.items():
            individual_metrics[model_name] = {
                'mae': mean_absolute_error(y_true, pred),
                'r2': r2_score(y_true, pred),
                'mape': np.mean(np.abs((y_true - pred) / (y_true + 1e-6))) * 100
            }
        
        # Performance improvement
        best_individual = min(individual_metrics.values(), key=lambda x: x['mae'])
        improvement = {
            'mae_improvement': (best_individual['mae'] - ensemble_metrics['mae']) / best_individual['mae'] * 100,
            'r2_improvement': (ensemble_metrics['r2'] - best_individual['r2']),
            'mape_improvement': (best_individual['mape'] - ensemble_metrics['mape']) / best_individual['mape'] * 100
        }
        
        result = {
            'ensemble': ensemble_metrics,
            'individual': individual_metrics,
            'improvement': improvement,
            'weights_used': weights
        }
        
        # Store performance
        self.ensemble_performance[category] = result
        
        return result