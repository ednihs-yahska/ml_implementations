import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

print("\n\n>>>>> Assuming data files are stored in a folder called hw3-data <<<<<<<< \n\n")


print("Running Naive...")
isAllNums = np.vectorize(lambda a, _: np.isreal(a))
isAnyNan = np.vectorize(lambda a, _: np.isnan(a))
isAnyInfinite = np.vectorize(lambda a, _: np.isinfinite(a))
fillNan = np.vectorize(lambda a, filler: filler if np.isnan(a) else a)
fillNanCat = np.vectorize(lambda a, _: str(a))

min_max_scaler_x = MinMaxScaler()
min_max_scaler_y = MinMaxScaler()

scale = True

train_data = pd.read_csv("hw3-data/my_train.csv")
dev_data = pd.read_csv("hw3-data/my_dev.csv")
test_data = pd.read_csv("hw3-data/test.csv")

columns = list(train_data)
feature_columns = np.array(columns)[1:-1]

train = np.array(train_data)
dev = np.array(dev_data)
test = np.array(test_data)

X_train = train[:,1:-1]
X_dev = dev[:, 1: -1]
X_test = test[:, 1:]

Y_train = train[:,-1]
Y_dev = dev[:, -1]

nums_override = [x for x in range(X_train.shape[1])]

def removeNans(D, _nums_override, _numfields=None, filler=0):
    if _numfields is not None:
        for c in range(D.shape[1]):
            if c in _numfields:
                median = np.median(D[:, c])
                median = filler if filler != -1 else median
                D[:, c] = fillNan(D[:, c], filler)
            else:
                D[:, c] = fillNanCat(D[:, c], -1)
    else:
        _numfields = []
        is_nums = isAllNums(D, -1)
        for c in range(D.shape[1]):
            if np.all(is_nums[:,c]) and c not in _nums_override:
                _numfields.append(c);
                median = np.median(D[:, c])
                median = filler if filler != -1 else median
                D[:, c] = fillNan(D[:, c], filler)
            else:
                D[:, c] = fillNanCat(D[:, c], -1)
    return D, _numfields

def processData(D, _nums_override, encoder = None, _numfields = None):
    if _numfields is not None:
        D, xt_num_fields = removeNans(D, _nums_override, _numfields, filler=-1)
    else:
        D, xt_num_fields = removeNans(D, _nums_override, filler=-1)
    xt_cat_fields = np.setdiff1d(np.array([x for x in range(D.shape[1])]), xt_num_fields)
    if encoder:
        xt_cat1hot = encoder.transform(D[:, xt_cat_fields])
    else:
        encoder = OneHotEncoder(categories="auto", handle_unknown="ignore")
        xt_cat1hot = encoder.fit_transform(D[:, xt_cat_fields])
    _1 = D[:, xt_num_fields]
    _2 = xt_cat1hot.toarray()
    D = np.append(_1, _2, axis=1)
    return D, encoder, xt_num_fields

X_train, cat_encoder, num_fields = processData(X_train, nums_override)

result_dev = processData(X_dev, nums_override, cat_encoder, num_fields)
X_dev = result_dev[0]

result_test = processData(X_test, nums_override, cat_encoder, num_fields)
X_test = result_test[0]

if scale and len(num_fields) > 0:
    X_train[:, num_fields] = min_max_scaler_x.fit_transform(X_train[:, num_fields])
    X_dev[:, num_fields] = min_max_scaler_x.fit_transform(X_dev[:, num_fields])
    X_test[:, num_fields] = min_max_scaler_x.fit_transform(X_test[:, num_fields])
    Y_train = min_max_scaler_y.fit_transform(Y_train)
    Y_dev = min_max_scaler_y.fit_transform(Y_dev)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
p_dev = lin_reg.predict(X_dev)
p_test = lin_reg.predict(X_test)

ridge = Ridge(alpha=10)
ridge.fit(X_train, Y_train)
p_dev_fit = ridge.predict(X_dev)
p_test_fit = ridge.predict(X_test)

if scale and len(num_fields) > 0:
    p_dev = min_max_scaler_y.inverse_transform(p_dev);
    p_test = min_max_scaler_y.inverse_transform(p_test);
    Y_dev = min_max_scaler_y.inverse_transform(Y_dev);
    p_dev_fit = min_max_scaler_y.inverse_transform(p_dev_fit);
    p_test_fit = min_max_scaler_y.inverse_transform(p_test_fit);

e_dev = mean_squared_log_error(Y_dev, p_dev)
rmlse_dev = np.sqrt(e_dev)
print("Naive Binarization RMLSE without regularization ", rmlse_dev)

e_dev_ridge = mean_squared_log_error(Y_dev, p_dev_fit)
rmlse_dev_ridge = np.sqrt(e_dev_ridge)
rmlse_dev_ridge
print("Naive Binarization RMLSE without regularization ", rmlse_dev_ridge)

print("Running Smart...")
nums_override = [0, 16, 17, 18, 19, 58, 75, 76]
scale=True

train = np.array(train_data)
dev = np.array(dev_data)
test = np.array(test_data)

X_train = train[:,1:-1]
X_dev = dev[:, 1: -1]
X_test = test[:, 1:]

Y_train = train[:,-1]
Y_dev = dev[:, -1]

X_train, cat_encoder, num_fields = processData(X_train, nums_override, None, None)

result_dev = processData(X_dev, nums_override, cat_encoder, num_fields)
X_dev = result_dev[0]

result_test = processData(X_test, nums_override, cat_encoder, num_fields)
X_test = result_test[0]

if scale and len(num_fields) > 0:
    X_train[:, num_fields] = min_max_scaler_x.fit_transform(X_train[:, num_fields])
    X_dev[:, num_fields] = min_max_scaler_x.fit_transform(X_dev[:, num_fields])
    X_test[:, num_fields] = min_max_scaler_x.fit_transform(X_test[:, num_fields])
    Y_train = min_max_scaler_y.fit_transform(Y_train.reshape((Y_train.shape[0], 1)))
    Y_dev = min_max_scaler_y.fit_transform(Y_dev.reshape((Y_dev.shape[0], 1)))

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
p_dev = lin_reg.predict(X_dev)
p_test = lin_reg.predict(X_test)

ridge = Ridge(alpha=10)
ridge.fit(X_train, Y_train)
p_dev_fit = ridge.predict(X_dev)
p_test_fit = ridge.predict(X_test)

if scale and len(num_fields) > 0:
    p_dev = min_max_scaler_y.inverse_transform(p_dev);
    p_test = min_max_scaler_y.inverse_transform(p_test);
    Y_dev = min_max_scaler_y.inverse_transform(Y_dev);
    p_dev_fit = min_max_scaler_y.inverse_transform(p_dev_fit);
    p_test_fit = min_max_scaler_y.inverse_transform(p_test_fit);

e_dev = mean_squared_log_error(Y_dev, p_dev)
rmlse_dev = np.sqrt(e_dev)
print("Smart Binarization RMLSE ", rmlse_dev)

e_dev_ridge = mean_squared_log_error(Y_dev, p_dev_fit)
rmlse_dev_ridge = np.sqrt(e_dev_ridge)
print("Smart Binarization RMLSE with regularization ", rmlse_dev_ridge)

print("Running Combinations...")

def combinations(X):
    add = X[:, 19] - X[:, 18]
    add = add.reshape(-1, 1)
    X_add = np.hstack([X, add])
    areas = [37, 42, 43, 45, 61, 65, 66, 67, 68, 69, 70]
    X[:, areas]
    area = X[:, 37] + X[:, 42] + X[:, 43] + X[:, 45] + X[:, 61] + X[:, 65] + X[:, 66] + X[:, 67] + X[:, 68] + X[:, 69] + X[:, 70]
    area = area.reshape(-1, 1)
    X_add = np.hstack([X_add, area])
    area_sq = area*area
    lot_area_sq = X[:, 3] * X[:, 3]
    lot_area_sq = lot_area_sq.reshape(-1, 1)
    X_add = np.hstack([X_add, area_sq, lot_area_sq])
    return X_add

nums_override = [0, 16, 17, 18, 19, 58, 75, 76]
scale=True

train = np.array(train_data)
dev = np.array(dev_data)
test = np.array(test_data)

X_train = train[:,1:-1]
X_dev = dev[:, 1: -1]
X_test = test[:, 1:]

Y_train = train[:,-1]
Y_dev = dev[:, -1]

X_train_add = combinations(X_train)
X_dev_add = combinations(X_dev)
X_test_add = combinations(X_test)

X_train, cat_encoder, num_fields = processData(X_train_add, nums_override, None, None)

result_dev = processData(X_dev_add, nums_override, cat_encoder, num_fields)
X_dev = result_dev[0]

result_test = processData(X_test_add, nums_override, cat_encoder, num_fields)
X_test = result_test[0]

if scale and len(num_fields) > 0:
    X_train[:, num_fields] = min_max_scaler_x.fit_transform(X_train[:, num_fields])
    X_dev[:, num_fields] = min_max_scaler_x.fit_transform(X_dev[:, num_fields])
    X_test[:, num_fields] = min_max_scaler_x.fit_transform(X_test[:, num_fields])
    Y_train = min_max_scaler_y.fit_transform(Y_train.reshape((Y_train.shape[0], 1)))
    Y_dev = min_max_scaler_y.fit_transform(Y_dev.reshape((Y_dev.shape[0], 1)))

ridge = Ridge()
ridge.fit(X_train, Y_train)
p_dev_fit = ridge.predict(X_dev)
p_test_fit = ridge.predict(X_test)

if scale and len(num_fields) > 0:
    p_dev = min_max_scaler_y.inverse_transform(p_dev.reshape(-1, 1));
    p_test = min_max_scaler_y.inverse_transform(p_test.reshape(-1, 1));
    Y_dev = min_max_scaler_y.inverse_transform(Y_dev.reshape(-1, 1));
    p_dev_fit = min_max_scaler_y.inverse_transform(p_dev_fit.reshape(-1, 1));
    p_test_fit = min_max_scaler_y.inverse_transform(p_test_fit.reshape(-1, 1));

e_dev_ridge = mean_squared_log_error(Y_dev, p_dev_fit)
rmlse_dev_ridge = np.sqrt(e_dev_ridge)
print("RMLSE for combination of features and quadratic features ", rmlse_dev_ridge)
