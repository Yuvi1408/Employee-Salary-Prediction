import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import median_abs_deviation
import warnings
import pickle
warnings.filterwarnings('ignore')


class Data:
    def __init__(self, train_features, train_target, test_features, target_col):
        self.train_features = train_features
        self.train_target = train_target
        self.test_features = test_features
        self.target_col = target_col
        self.invalid_flag = False
        self.process_data()

    def process_data(self):
        self._create_train_df()
        self._create_test_df()
        self._column_info()
        self._flag_invalids(self.train_df, self.target_col)
        self._check_duplicates()

    def _create_train_df(self):
        train_feature_df = pd.read_csv(self.train_features)
        train_target_df = pd.read_csv(self.train_target)
        self.train_df = pd.merge(train_feature_df, train_target_df, left_index=True, right_index=True)

    def _create_test_df(self):
        self.test_df = pd.read_csv(self.test_features)

    def _column_info(self):
        self.cat_cols = self.train_df.select_dtypes(include=['O']).columns.tolist()
        self.num_cols = self.train_df.select_dtypes(exclude=['O']).columns.tolist()

    def _flag_invalids(self, df, col):
        if np.sum(df[col] <= 0) > 0:
            self.invalid_flag = True

    def _check_duplicates(self):
        self.train_df.drop_duplicates(inplace=True)
        self.test_df.drop_duplicates(inplace=True)


class Plots:
    sns.set(style="darkgrid")

    def __init__(self, data):
        self.train_df = data.train_df
        self.target_col = data.target_col
        self.cat_cols = data.cat_cols
        self.num_cols = data.num_cols
        self.eda_process()

    def eda_process(self):
        self._distplot()
        self._boxplot()
        self._heatmap()

    def _distplot(self):
        fig = plt.figure(figsize=(14, 14))
        for idx, col in enumerate(self.num_cols):
            fig.add_subplot(len(self.num_cols), 1, idx + 1)
            sns.histplot(self.train_df[col], bins=20, color='g', kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()

    def _boxplot(self):
        df = self.train_df.copy()
        fig = plt.figure(figsize=(14, 18))
        for idx, col in enumerate(self.cat_cols):
            if df[col].nunique() < 10:
                fig.add_subplot(3, 2, idx + 1)
                sns.boxplot(x=col, y=self.target_col, data=df)
                plt.title(f'Salary vs {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()

    def _heatmap(self):
        df = self.train_df.copy()
        for col in self.cat_cols:
            if df[col].nunique() < 100:
                df[col + '_mean'] = df.groupby(col)[self.target_col].transform('mean')
        corr_cols = [col for col in df.columns if 'mean' in col or col in ['yearsExperience', 'milesFromMetropolis', 'salary']]
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='Purples')
        plt.title('Correlation Heatmap')
        plt.tight_layout()


class FeatEng:
    def __init__(self, data):
        self.data = data
        self.invalid_flag = data.invalid_flag
        self.target = data.target_col
        self.cat_cols = ['jobType', 'degree', 'major', 'industry', 'companyId']
        self.input_cols = ['jobType', 'industry', 'degree', 'major']
        self.labels = {}
        self._process_feat_eng()

    def _process_feat_eng(self):
        self._feat_eng_train_df()
        self._feat_eng_test_df()

    def _feat_eng_train_df(self):
        if self.invalid_flag:
            self._clean_df()
        df = self._encode_labels(self.data.train_df, self.cat_cols)
        if 'jobId' in df.columns:
            df = df.drop(columns='jobId')
        group_df = self._group_stats(df)
        self.data.train_df = pd.merge(df, group_df, on=self.cat_cols, how='left')

    def _feat_eng_test_df(self):
        df = self._encode_labels(self.data.test_df, self.cat_cols, test_data=True)
        if 'jobId' in df.columns:
            df = df.drop(columns='jobId')
        df = pd.merge(df, self.group_df, on=self.cat_cols, how='left')
        self.data.test_df = self._replace_nan(df)

    def _encode_labels(self, df, cols, test_data=False):
        for col in cols:
            if not test_data:
                le = LabelEncoder()
                le.fit(df[col])
                self.labels[col] = le
                df[col] = le.transform(df[col])
            else:
                df[col] = df[col].map(lambda x: self.labels[col].transform([x])[0] if x in self.labels[col].classes_ else -1)
        return df

    def _clean_df(self):
        self.data.train_df = self.data.train_df[self.data.train_df['salary'] > 0].reset_index(drop=True)

    def _group_stats(self, df):
        group_df = df.groupby(self.cat_cols).agg(
            group_mean=(self.target, 'mean'),
            group_median=(self.target, 'median'),
            group_max=(self.target, 'max'),
            group_min=(self.target, 'min'),
            group_mad=(self.target, median_abs_deviation)
        ).reset_index()
        self.group_df = group_df
        return group_df

    def _replace_nan(self, df):
        for col in ['group_mean', 'group_median', 'group_max', 'group_min', 'group_mad']:
            df[col] = df[col].fillna(df.groupby(self.input_cols)[col].transform('mean'))
        return df


class ModelEvaluation:
    def __init__(self, data, models):
        self.data = data
        self.models = models
        self.mse = {}
        self.best_model = None
        self.target_col = data.target_col

        object_cols = self.data.train_df.select_dtypes(include='object').columns
        if len(object_cols) > 0:
            print("Dropping object columns:", object_cols.tolist())
            self.data.train_df.drop(columns=object_cols, inplace=True, errors='ignore')
            self.data.test_df.drop(columns=object_cols, inplace=True, errors='ignore')

        self.base_target_df = data.train_df['group_mean']
        self.target_df = data.train_df[self.target_col]
        self.feature_df = data.train_df.drop(columns=[self.target_col])
        self.test_df = data.test_df
        self._process_models()

    def _process_models(self):
        self._baseline_model()
        self._cross_validate_model()
        self._best_model_process()

    def _baseline_model(self):
        self.mse['Baseline_model'] = mean_squared_error(self.target_df, self.base_target_df)

    def _cross_validate_model(self, cv=3):
        for model in self.models:
            score = cross_val_score(model, self.feature_df, self.target_df,
                                    scoring='neg_mean_squared_error', cv=cv)
            self.mse[model] = -score.mean()

    def _best_model_process(self):
        self.best_model = min(self.mse, key=self.mse.get)
        self.best_model.fit(self.feature_df, self.target_df)
        self.test_df[self.target_col] = self.best_model.predict(self.test_df)
        self._plot_feature_importance()

    def _plot_feature_importance(self):
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)
            features = self.feature_df.columns
            plt.figure(figsize=(8, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center', color='r')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.tight_layout()
            plt.show()

    @staticmethod
    def save_results(filepath, df):
        df.to_csv(filepath, index=False)

    @staticmethod
    def save_best_model(filepath, model):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def tune_hyperparameter(estimator, param_grid, feature_df, target_df, n_iter=5):
        rs_cv = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=n_iter,
                                   scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
        rs_cv.fit(feature_df, target_df)
        return rs_cv.best_params_


# ----------------- RUN PIPELINE ------------------

train_features = 'data/train_features.csv'
train_target = 'data/train_salaries.csv'
test_features = 'data/test_features.csv'
target_col = 'salary'

data = Data(train_features, train_target, test_features, target_col)
visuals = Plots(data)
feature_engineering = FeatEng(data)

lr = LinearRegression()
rfr = RandomForestRegressor(n_estimators=60, max_depth=15, min_samples_split=80, max_features=8, n_jobs=4)
gboost = GradientBoostingRegressor(n_estimators=40, max_depth=7, loss='squared_error')  # FIXED

models = [lr, rfr, gboost]
model_eval = ModelEvaluation(data, models)

model_eval.save_results('./models/Salary_Prediction.csv', model_eval.test_df)
model_eval.save_best_model('./models/Salary_Prediction_Best_Model.sav', model_eval.best_model)
