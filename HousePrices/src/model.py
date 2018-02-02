import pandas as pd
import numpy as np
from scipy.stats import norm, skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import time
from scipy.stats import randint

def cal_missing_ratio(data):
    all_data_na = (data.isnull().sum() / len(data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data)

n_folds = 5
def rmsle_cv(model, data, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data)
    rmse= np.sqrt(-cross_val_score(model, data, y_train, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))
    return(rmse)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                                            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    test["SalePrice"] = 0
    test_ID = test['Id']

    # Deleting outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    y = train.SalePrice
    train_rows = train.shape[0]
    data = pd.concat((train, test)).reset_index(drop=True)


    filling_in_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
                       'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'MasVnrType', 'MSSubClass']

    filling_in_0 = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

    filling_in_mode = ['MSZoning', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

    for col in (filling_in_none):
        data[col] = data[col].fillna('None')

    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    for col in (filling_in_0):
        data[col] = data[col].fillna(0)

    for col in (filling_in_mode):
        data[col] = data[col].fillna(data[col].mode()[0])

    drop_feature = ['Id', 'Utilities', 'SalePrice']
    data.drop(drop_feature, axis=1, inplace=True)


    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['OverallCond'] = data['OverallCond'].astype(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    from sklearn.preprocessing import LabelEncoder
    factorize_col = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')

    for c in factorize_col:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    #box cox
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        data[feat] = boxcox1p(data[feat], lam)

    data = pd.get_dummies(data)
    train = data[:train_rows]
    test = data[train_rows:]


    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(range(50,1000,10), grid_search.cv_results_['mean_test_score'])
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(range(50,1000,10), grid_search.cv_results_['std_test_score'])
    # plt.show()


    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    # GBoost = GradientBoostingRegressor(n_estimators=480, learning_rate=0.05, max_depth=3, max_features=26,
    #                                    min_samples_leaf=7, min_samples_split=15, loss='huber', random_state=5,
    #                                    subsample=0.8)

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)


    # model_xgb = xgb.XGBRegressor(n_estimators=3000,max_depth=3,min_child_weight=3,gamma=0,subsample=0.95,
    #         colsample_bytree=0.6,reg_alpha=1e-6,learning_rate=0.01,silent=1, random_state=7, nthread=-1)

    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=7, nthread=-1)


    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)



    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    stacked_averaged_models.fit(train.values, y.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

    model_xgb.fit(train, y)
    xgb_pred = np.expm1(model_xgb.predict(test))

    model_lgb.fit(train, y)
    lgb_pred = np.expm1(model_lgb.predict(test.values))

    ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('../data/submission.csv', index=False)