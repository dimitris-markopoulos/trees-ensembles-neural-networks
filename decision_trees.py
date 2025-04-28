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

        model = DecisionTreeClassifier(random_state=1234)
        model.fit(self.X_tr, self.y_tr)

        y_tr_pred = model.predict(self.X_tr)
        y_pred = model.predict(self.X_ts)
        params = model.get_params()

        keys_to_pull = ['max_depth', 'min_samples_leaf'] # List of keys you want
        filtered_dict = {k: params[k] for k in keys_to_pull}

        if print_:
            print(f'===== Default Model =====')
            print(f'Training Accuracy: {accuracy_score(self.y_tr, y_tr_pred):.4f}')
            print(f'Testing Accuracy: {accuracy_score(self.y_ts, y_pred):.4f}')
            print(f'Default Params: {filtered_dict}')

        
        return {'model': model, 'y_tr_pred': y_tr_pred, 'y_pred': y_pred}

       
    # ===== tune model - grid search CV =====
    def tune_run(self, param_grid=None, balance_classes=True):

        if param_grid == None:
            param_grid = {
                'max_depth': [2, 3, 5, 7, 9, 11, 13, None],
                'min_samples_leaf': [1, 5, 10, 20, 50],
                # 'min_samples_split': [2, 10],
                # 'max_features': [None, 'sqrt'],
                # 'criterion': ['gini', 'entropy']
            }

        if balance_classes == True:
            class_weight_setting = 'balanced'
        else:
            class_weight_setting = None

        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=1234, class_weight=class_weight_setting),
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
        best_params = grid.best_params_
        self.best_params = best_params
        best_score = grid.best_score_
        self.best_score = best_score
        self.balance_classes = balance_classes
    
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
        axes[1].set_title('Validation')
        axes[1].set_xlabel('min_samples_leaf')
        axes[1].set_ylabel('max_depth')

        plt.suptitle(f'Decision Tree - Accuracy | Grid of Hyperparams | Balanced = {self.balance_classes}')
        plt.savefig('media/02_decision_tree_overfitting_viz')
        plt.show()

    def best_model(self, print_=True):

        best_estimator = self.grid.best_estimator_
        y_tr_pred = best_estimator.predict(self.X_tr)
        y_pred = best_estimator.predict(self.X_ts)

        if print_:
            print(f'===== Tuned Model =====')
            print(f'Training Accuracy: {accuracy_score(self.y_tr, y_tr_pred):.4f}')
            print(f'Testing Accuracy: {accuracy_score(self.y_ts, y_pred):.4f}')
            print(f'Validation (CV) Accuracy: {self.best_score:.4f}') 
            print(f'Best params: {self.best_params}')

        return self.best_params
