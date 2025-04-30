**WORK IN PROGRESS**

# Trees & Ensemble Models – UCI Adult Dataset

This project explores decision trees and ensemble methods (Random Forest, AdaBoost) on the UCI Adult Income dataset to analyze model behavior, overfitting tendencies, and feature importances.

## 📁 Dataset
- Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Task: Binary classification – Predict whether income >$50K based on demographic features

## Models Implemented
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **Neural Networks** (Feedforward MLP)

## Objectives
1. **Overfitting Exploration**: Visualize how key hyperparameters (e.g., `max_depth`, `n_estimators`, `learning_rate`) affect model overfitting using validation heatmaps.
2. **Hyperparameter Tuning**: Use `GridSearchCV` to optimize each model’s performance.
3. **Interpretability**:
   - Plot classification trees for representative models
   - Visualize and compare top feature importances across models

## Visual Outputs
- Validation curves for each model across many hyperparameters
- Overfitting maps (train vs. validation accuracy gap)
- Top 10 feature importances per model

## Interpretation
Compare and contrast how each model overfits, tunes, and ranks features.  
Do models identify similar top features? Discuss the consistency and meaning of those rankings.



