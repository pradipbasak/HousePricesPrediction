# Loading the dataset
import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = pd.concat([train, test], axis=0).drop("Id", axis=1)

# Finding the no. of null values in each feature
null_values = []
for col in df.columns:
    null_values.append([col, df[col].isnull().sum()])


SalePrice = df["SalePrice"]
df.drop(["Alley", "Fence", "FireplaceQu", "MiscFeature", "PoolQC", "SalePrice"], inplace=True, axis=1)

categorical_features = ["BldgType", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                        "BsmtFinType2", "BsmtQual", "CentralAir", "Condition1", "Condition2", 
                        "Electrical", "ExterCond", "ExterQual", "Exterior1st", "Exterior2nd", 
                        "Foundation", "Functional", "GarageCond", 
                        "GarageFinish", "GarageQual", "GarageType", "Heating", "HeatingQC", 
                        "HouseStyle", "KitchenQual", "LandContour", "LandSlope", "LotConfig", 
                        "LotShape", "MSZoning", "MasVnrType", "Neighborhood", "PavedDrive", 
                        "RoofMatl", "RoofStyle", "SaleCondition", "SaleType", "Street", 
                        "Utilities"]

median_features = ["BsmtFinSF1", "BsmtFinSF2", "BsmtFullBath", "BsmtHalfBath", "BsmtUnfSF", "GarageArea", 
                   "GarageCars", "GarageYrBlt", "LotFrontage", "MasVnrArea", "TotalBsmtSF"]

# Handling missing values
for item in df.columns:
    if item in categorical_features:
        df[item] = df[item].fillna("unkwown")
    elif item in median_features:
        df[item] = df[item].fillna(df[item].median())
    else:
        df[item] = df[item].fillna(df[item].mode())
        
#null_cols = df.isnull().sum()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
for item in df.columns:
    if item in categorical_features:
        le = LabelEncoder()
        df[item] = le.fit_transform(df[item])

# Split into X and y
df = pd.concat([df, SalePrice], axis=1)
X = pd.DataFrame(columns=df.columns)
X = df.loc[df["SalePrice"].notnull()]
y = X["SalePrice"]
X = X.drop("SalePrice", axis=1)

# XGBoost
"""import xgboost
model = xgboost.XGBRegressor()
booster=["gbtree","gblinear", "dart"]
base_score=[0.25, 0.5, 0.75, 1] 
n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]
max_depth = [2, 3, 5, 10, 15]
learning_rate=[0.05, 0.1, 0.15, 0.20, 0.25]
min_child_weight=[1, 2, 3, 4, 5]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_estimators': n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate,
                       'min_child_weight':min_child_weight, 'booster':booster, 'base_score':base_score}"""
                       
# Final Model
import xgboost
model = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=3, min_child_weight=4, missing=None, n_estimators=500,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

# Accuracy
def accuracy(y_true, y_pred):
    error = 0
    for i in range(len(y_true)):
        error += ((y_true.iloc[i] - y_pred.iloc[i])/y_true.iloc[i]) ** 2
    return error/len(y_true)

# KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=0) 

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test))
    print("Loss = %.3f" % (accuracy(y_test, y_pred)))

# Storing the final result
X_test = df.loc[df["SalePrice"].isnull()].drop("SalePrice", axis=1)
y_res = model.predict(X_test)
ids = test["Id"]
result = pd.DataFrame(columns=["Id", "SalePrice"])
result = pd.concat([pd.DataFrame(ids), pd.DataFrame(y_res)], axis=1)
result.to_csv("result.csv", index=False)