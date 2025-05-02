import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

class FFNN:
    def __init__(self, X_tr, X_ts, y_tr, y_ts):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts

    def grid_search_CV(self, param_grid, cv=2, epochs=500, early_stopping=True, validation_fraction=0.1, verbose=1):
        mlp = MLPClassifier(max_iter=epochs, early_stopping=early_stopping, validation_fraction=validation_fraction)
        start = time.time()
        self.grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            return_train_score=True,
            verbose=verbose
        )
        self.grid_search.fit(self.X_tr, self.y_tr.values.ravel())
        print(f"Elapsed Time: {time.time() - start:.2f} seconds")

    def random_search_CV(self, param_dist, cv=2, n_iter=50, epochs=500, early_stopping=True, validation_fraction=0.1, verbose=1):
        mlp = MLPClassifier(max_iter=epochs, early_stopping=early_stopping, validation_fraction=validation_fraction)
        start = time.time()
        self.random_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=verbose,
            return_train_score=True
        )
        self.random_search.fit(self.X_tr, self.y_tr.values.ravel())
        print(f"Elapsed Time: {time.time() - start:.2f} seconds")

    def extract_cv_results(self, search):
        df = pd.DataFrame(search.cv_results_)
        results_df = pd.DataFrame({
            'params': search.cv_results_['params'],
            'mean_train_score': search.cv_results_['mean_train_score'],
            'mean_test_score': search.cv_results_['mean_test_score']
        })
        return df, results_df

    def plot_validation_hyperparam_grid_row(self, results_df, param, ax, colors=['purple', 'black']):
        scores = ['mean_test_score', 'mean_train_score']
        param_df = results_df['params'].apply(lambda x: x.get(param, None)).to_frame(name=param)
        plot_df = pd.concat([param_df, results_df[scores]], axis=1)
        grouped = plot_df.groupby(param)[scores].mean().reset_index().sort_values(param)
        x_vals = grouped[param].astype(str)
        ax.plot(x_vals, grouped['mean_test_score'], label='Validation', color=colors[0])
        ax.plot(x_vals, grouped['mean_train_score'], label='Training', color=colors[1])
        ax.scatter(x_vals, grouped['mean_test_score'], color=colors[0], alpha=0.5)
        ax.scatter(x_vals, grouped['mean_train_score'], color=colors[1], alpha=0.5)
        ax.set_title(f"FFNN | {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Accuracy")

    def plot_all_validation_curves(self, search, param_list, save_path=None):
        _, results_df = self.extract_cv_results(search)
        fig, axes = plt.subplots(1, len(param_list), figsize=(5 * len(param_list), 4))
        for ax, param in zip(axes, param_list):
            self.plot_validation_hyperparam_grid_row(results_df, param, ax)
        axes[0].legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def report_test_performance(self, search, model_name="FFNN"):
        best_mlp = search.best_estimator_
        y_pred = best_mlp.predict(self.X_ts)

        test_acc = accuracy_score(self.y_ts, y_pred)
        train_acc = accuracy_score(self.y_tr, best_mlp.predict(self.X_tr))
        val_acc = search.best_score_

        param_keys = list(search.best_params_.keys())
        print(f"{model_name} - Hyperparameters: {param_keys}")
        print(f"Train Accuracy: {train_acc * 100:.3f}%")
        print(f"CV Validation Accuracy: {val_acc * 100:.3f}%")
        print(f"Test Accuracy: {test_acc * 100:.3f}%")
        print("Best Hyperparameters:", search.best_params_)

