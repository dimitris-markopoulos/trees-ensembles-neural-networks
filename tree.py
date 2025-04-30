# ===== import libraries ======
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.base import clone

class TreeModel:
    def __init__(self, X_tr, X_ts, y_tr, y_ts, model, name='Model'):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts
        self.base_model = model
        self.name = name

    def fit_predict(self, param_grid, balance_classes=True, k_folds=5, scoring='accuracy'):
        self.param_grid = param_grid 
        self.hyper_grid = list(self.param_grid.keys())

        if balance_classes == True:
            self.base_model.set_params(class_weight='balanced')
        else:
            self.base_model.set_params(class_weight=None)

        grid = GridSearchCV(
            estimator=self.base_model,
            param_grid=param_grid,
            cv=k_folds,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        fit = grid.fit(self.X_tr, self.y_tr)
        results_df = pd.DataFrame(grid.cv_results_)

        y_tr_pred = grid.best_estimator_.predict(self.X_tr) # predict using training data (to see training results)
        y_pred = grid.best_estimator_.predict(self.X_ts)    # typical procedure - predict using testing data

        self.grid = grid
        self.results_df = results_df
        self.y_tr_pred = y_tr_pred
        self.y_ts_pred = y_pred
        self.tr_accuracy = accuracy_score(self.y_tr, y_tr_pred)
        self.ts_accuracy = accuracy_score(self.y_ts, y_pred)
        self.cv_validation_accuracy = grid.best_score_
        self.tuned_params = grid.best_params_
    
    @property
    def tr_ts_scoring(self):
        print(f'{str(self.base_model)}')
        print(f'Grid - Hyperparameters: {self.hyper_grid}')
        print(f'Train Accuracy: {self.tr_accuracy*100:.3f}%')
        print(f'CV Validation Accuracy: {self.cv_validation_accuracy*100:.3f}%')
        print(f'Test Accuracy: {self.ts_accuracy*100:.3f}%')

    def plot_validation_hyperparam_grid_row(self, param_list, figsize=(5, 4), colors=['purple', 'black'], save_path=None):
        n_params = len(param_list)
        df = self.results_df.copy()

        if n_params == 1:

            param = param_list[0]
            scores = ['mean_test_score', 'mean_train_score']
            param_df = df['params'].apply(lambda x: x[param])
            results_df = pd.concat([param_df, df[scores]], axis=1).rename(columns={'params':param})
            grouped = results_df.groupby(param)[scores].mean().reset_index(drop=False)
            params = grouped[param]
            ts_scores, tr_scores = grouped[scores[0]], grouped[scores[1]]

            plt.figure(figsize=figsize)
            plt.plot(params, ts_scores, label = 'Validation Set', color = colors[0])
            plt.plot(params, tr_scores, label = 'Training Set', color = colors[1])
            plt.scatter(params, ts_scores, color=colors[0], alpha = 0.5)
            plt.scatter(params, tr_scores, color=colors[1], alpha = 0.5)
            plt.xlabel(param)
            plt.ylabel('Accuracy')
            plt.title(f'GridsearchCV - Validation Curve | {self.name}')
            plt.legend()
            plt.savefig(save_path)
            plt.show()

        else:
            n_params = len(param_list)
            fig, axes = plt.subplots(1, n_params, figsize=(figsize[0]*n_params, figsize[1]), constrained_layout=True)
            
            for i, param in enumerate(param_list):
                ax = axes[i] if n_params > 1 else axes

                scores = ['mean_test_score', 'mean_train_score']
                param_df = df['params'].apply(lambda x: x[param])
                results_df = pd.concat([param_df, df[scores]], axis=1).rename(columns={'params': param})
                grouped = results_df.groupby(param)[scores].mean().reset_index(drop=False)

                x_vals = grouped[param]
                ts_scores = grouped[scores[0]]
                tr_scores = grouped[scores[1]]

                ax.plot(x_vals, ts_scores, label='Validation', color=colors[0])
                ax.plot(x_vals, tr_scores, label='Training', color=colors[1])
                ax.scatter(x_vals, ts_scores, color=colors[0], alpha=0.5)
                ax.scatter(x_vals, tr_scores, color=colors[1], alpha=0.5)
                ax.set_xlabel(param)
                ax.set_ylabel("Accuracy")
                ax.set_title(param.replace('_', ' '))

                if i == 0:
                    ax.legend(loc='best')

            fig.suptitle(f'Validation Curves | {self.name}', fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.show()

    def viz_heatmap_cv(self, top_two_params_for_overfitting):
        df = self.results_df.copy()
        param1, param2 = top_two_params_for_overfitting
        df[param1] = df['params'].apply(lambda x: x[param1])
        df[param2] = df['params'].apply(lambda x: x[param2])
        train = df.pivot_table(index=param1, columns=param2, values='mean_train_score')
        test = df.pivot_table(index=param1, columns=param2, values='mean_test_score')
        diff = train - test

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        sns.heatmap(train, annot=True, fmt=".3f", cmap='viridis', ax=axes[0])
        axes[0].set_title('Training Accuracy')
        sns.heatmap(test, annot=True, fmt=".3f", cmap='viridis', ax=axes[1])
        axes[1].set_title('Validation Accuracy')
        sns.heatmap(diff, annot=True, fmt=".3f", cmap='coolwarm', center=0, ax=axes[2])
        axes[2].set_title('Overfitting (Train - Val)')
        filename = f'media/{self.name.lower().replace(" ", "_")}_heatmap_{param1}_{param2}.png'
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(f'{self.name} | Accuracy Grid | {param1} vs {param2}')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()