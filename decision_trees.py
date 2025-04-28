# Decision Trees

# ===== import libraries ======
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

class DecisionTree:

    def __init__(self, X_tr, X_ts, y_tr, y_ts):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts

    # ===== default model -> OVERFIT =====
    def default_decision_tree(self, print_=True):
        default_params = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'class_weight': None
        }

        model = DecisionTreeClassifier(random_state=1234, **default_params)
        model.fit(self.X_tr, self.y_tr)

        y_tr_pred = model.predict(self.X_tr)
        y_pred = model.predict(self.X_ts)

        if print_:
            print(f'===== Default Model =====')
            print(f'Training Accuracy: {accuracy_score(self.y_tr, y_tr_pred):.4f}')
            print(f'Testing Accuracy: {accuracy_score(self.y_ts, y_pred):.4f}')

        return {'model': model, 'y_tr_pred': y_tr_pred, 'y_pred': y_pred}

       
    # ===== tune model - grid search CV =====
    def tune_run(self, param_grid = None):

        if param_grid == None:
            param_grid = {
                'max_depth': [2, 3, 5, 7, 9, 11, 13, None],
                'min_samples_leaf': [1, 5, 10, 20, 50],
            }

        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=1234, class_weight='balanced'),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            return_train_score=True
        )

        grid.fit(self.X_tr, self.y_tr);

        results_df = pd.DataFrame(grid.cv_results_)

        self.results_df = results_df
        self.grid = grid
    
    # ===== Viz - hyperparams vs training and testing error =====
    @property
    def viz_cv(self):
        df1 = self.results_df.copy()
        df1['param_max_depth'] = df1['param_max_depth'].apply(lambda x: 'None' if x is None else int(x))
        df1['param_min_samples_leaf'] = df1['param_min_samples_leaf'].astype(int)

        heatmap_data_tr = df1.pivot_table(
            index='param_max_depth',
            columns='param_min_samples_leaf',
            values='mean_train_score'
        )

        heatmap_data_ts = df1.pivot_table(
            index='param_max_depth',
            columns='param_min_samples_leaf',
            values='mean_test_score'
        )

        fig, axes = plt.subplots(1,2,figsize=(16,6))
        sns.heatmap(heatmap_data_tr, annot=True, fmt=".3f", cmap='viridis', ax=axes[0])
        axes[0].set_title('Training')
        axes[0].set_xlabel('min_samples_leaf')
        axes[0].set_ylabel('max_depth')
        sns.heatmap(heatmap_data_ts, annot=True, fmt=".4f", cmap='viridis', ax=axes[1])
        axes[1].set_title('Testing')
        axes[1].set_xlabel('min_samples_leaf')
        axes[1].set_ylabel('max_depth')

        plt.suptitle('Decision Tree - Accuracy | Grid of Hyperparams')
        plt.savefig('media/02_decision_tree_overfitting_viz')
        plt.show()

    def best_model(self, print_=True):
        best_params = self.grid.best_params_
        best_score = self.grid.best_score_

        best_estimator = self.grid.best_estimator_
        y_tr_pred = best_estimator.predict(self.X_tr)
        y_pred = best_estimator.predict(self.X_ts)

        if print_:
            print(f'===== Tuned Model =====')
            print(f'Training Accuracy: {accuracy_score(self.y_tr, y_tr_pred):.4f}')
            print(f'Testing Accuracy: {accuracy_score(self.y_ts, y_pred):.4f}')
            print(f'Validation (CV) Accuracy: {best_score:.4f}') 
            print(f'Best params: {best_params}')

        return {'best params': best_params, 'best score': best_score}
