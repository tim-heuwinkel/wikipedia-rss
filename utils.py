import math
from typing import List

import numpy as np
import pandas as pd
from geopy import distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.model_selection import KFold, train_test_split
from tqdm.notebook import tqdm


def read_txt(f_path) -> List[List]:
    with open(f_path) as f:
        lines = f.read().splitlines()
        lines_sep = [line.split("\t") for line in lines]
    return lines_sep


def get_closest_city(coords, df_cities):
    """Returns closest city to given coords along with distance."""

    closest_dist = 0.0
    closest_city = ""
    for index, city in df_cities.iterrows():
        dist = distance.great_circle(coords, city["center_lat":"center_long"]).m
        if dist < closest_dist:
            closest_dist = dist
            closest_city = city["name"]

    return [closest_city, closest_dist]
    

def _find_coord(x, df):
    """Returns id, latitude and longitude for property with given id"""

    _id, lat, long = x[0], x[1], x[2]
    row = df[df["_id"] == _id].iloc[0]
    return row["_id"], row["latitude"], row["longitude"]


def add_dummy_category(structured):
    """Returns structured data with org_category as dummy features."""

    if "article_count" in structured.columns:
        structured.drop(["article_count"], axis=1, inplace=True)
    structured["org_category"] = [x.replace(" ", "_") for x in structured["org_category"]]
    dummies = pd.get_dummies(structured["org_category"], prefix="dummy")
    structured = pd.concat([structured, dummies], axis=1)
    structured.drop(["org_category"], axis=1, inplace=True)
    if not "_category" in structured.columns:
        structured.rename({"category": "_category"}, axis=1, inplace=True)

    return structured


def make_train_test(df, dummies=["MUNICODE"], verbose=True, test_size=0.25):
    """Returns train/test sets along with column names and df for saving errors"""

    to_drop = ["PROPERTYZIP", "MUNICODE", "SCHOOLCODE", "NEIGHCODE", "SALEDATE", "SALEPRICE",
               "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"]
    to_drop = [x for x in to_drop if not x in dummies]

    X = df.drop(to_drop, axis=1)

    if len(dummies) > 0:
        X = pd.get_dummies(X, columns=dummies)

    # save col names for later
    X_columns = list(X.columns)
    # remove id from col list, since it will be filtered out later
    X_columns.remove("_id")

    y = df["SALEPRICE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42)

    # remove unknown levels in test
    cols = X_train.columns
    for col in cols:
        if "_" in col and len(pd.unique(X_train[col])) <= 2:
            if np.sum(X_train[col]) == 0:  # no observations from this level
                if verbose:
                    print(f"removed column {col}, first occurence in test")
                X_train.drop([col], axis=1, inplace=True)
                X_train_train.drop([col], axis=1, inplace=True)
                X_train_val.drop([col], axis=1, inplace=True)
                X_test.drop([col], axis=1, inplace=True)
                X_columns.remove(col)

    # save ids for later
    train_ids = X_train["_id"]
    test_ids = X_test["_id"]
    X_train = X_train.drop(["_id"], axis=1)  # remove first column (id)
    X_test = X_test.drop(["_id"], axis=1)  # remove first column (id)

    if verbose:
        print("")
        print(f"{X_train.shape}: {X_train_train.shape} + {X_train_val.shape}")
        print(f"{y_train.shape}: {y_train_train.shape} + {y_train_val.shape}")
        print(X_test.shape)
        print(y_test.shape)

    # create error df
    error_df = pd.DataFrame(
        data={"id": test_ids, "lat": [0] * len(test_ids), "long": [0] * len(test_ids)})
    error_df = error_df.apply(lambda x: _find_coord(
        x, df), axis=1, result_type='broadcast')

    return X_columns, [X, y, X_train, X_test, y_train, y_test, X_train_train, X_train_val, y_train_train,
                       y_train_val], error_df


def mean_absolute_percentage_error(y_true, y_pred):
    """Returns MAPE"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def numeric_ndcg(y_true, y_pred, k=300):
    """Returns normalized discounted cumulative gain for integer scores."""

    # rank numeric label and create linearly decreasing relevance score
    y_true_rank = [sorted(y_true, reverse=True).index(x)+1 for x in y_true]
    y_len = len(y_true)
    relevance_true = [(y_len - y_true_rank[i] + 1) / y_len for i in range(y_len)]

    # format correctly
    relevance_true = np.asarray([relevance_true])
    y_pred = np.asarray([y_pred])

    return ndcg_score(relevance_true, y_pred, k=k)


def get_metrics(y_true, y_pred, print_out=True):
    """Returns MAE, RMSE and NDCG"""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    ndcg = numeric_ndcg(y_true, y_pred)

    if print_out:
        print(f"MAE:  {round(mae, 3)}")
        print(f"RMSE: {round(rmse, 3)}")
        print(f"NDCG:  {round(ndcg, 3)}")

    return mae, rmse, ndcg


def cross_validation(estimator, X, y, k_folds, additional_drops=[], verbose_drop=True, return_std=False):
    """Returns and prints cross validated MAE, RMSE, MAPE and R^2"""

    maes, rmses, ndcgs = [], [], []
    # fis = []
    X_cv = X.copy()
    X_cv_cols = X.columns

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X_cv), total=5):
        X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train.drop(additional_drops, axis=1, inplace=True)
        X_test.drop(additional_drops, axis=1, inplace=True)

        if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
            # impute with mean of train
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())

        # remove unknown levels in test
        X_cv_cols = list(X_train.columns)
        for col in X_cv_cols:
            if "_" in col and len(pd.unique(X_train[col])) <= 2:  # is binary column
                if np.sum(X_train[col]) == 0:  # no observations from this level
                    if verbose_drop:
                        print(f"removed column {col}, first occurence in test")
                    X_train.drop([col], axis=1, inplace=True)
                    X_test.drop([col], axis=1, inplace=True)
                    X_cv_cols.remove(col)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        if "linear_model" in str(type(estimator)):
            estimator.fit(X=X_train, y=y_train)
        else:
            estimator.fit(X=X_train, y=y_train)

        y_pred_cv = estimator.predict(X_test)
        mae, rmse, ndcg = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        ndcgs.append(ndcg)

    mae_cv, rmse_cv = round(np.mean(maes), 2), round(np.mean(rmses), 2)
    ndcg_cv = round(np.mean(ndcgs), 3)

    mae_std = round(math.sqrt(np.mean((maes - np.mean(maes))**2)), 2)
    rmse_std = round(math.sqrt(np.mean((rmses - np.mean(rmses))**2)), 2)
    ndcg_std = round(math.sqrt(np.mean((ndcgs - np.mean(ndcgs))**2)), 3)

    print("")
    print(f"MAE:  {mae_cv} \u00B1 {mae_std}")
    print(f"RMSE: {rmse_cv} \u00B1 {rmse_std}")
    print(f"NDCG: {ndcg_cv} \u00B1 {ndcg_std}")

    if return_std:
        return [mae_cv, rmse_cv, ndcg_cv], [mae_std, rmse_std, ndcg_std], X_cv_cols, # fis_cv
    else:
        return [mae_cv, rmse_cv, ndcg_cv], X_cv_cols, # fis_cv


def soos_validation(estimator, df, additional_drops=[], verbose_drop=True, standardize=False, return_std=False, split_var="_borough", verbose=True):
    """Spatial out-of-sample validation for given estimator and df. By default returns aggregated metrics for each borough."""

    soos_df = df.copy()
    soos_df = soos_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data
    soos_df.sort_values(by=[split_var])  # sort by borough

    error_df_soos = pd.DataFrame(
        data={"id": soos_df["venue_id"],
              "lat": soos_df["latitude"],
              "long": soos_df["longitude"],
              split_var: soos_df[split_var],
              "prediction": 0,
              "error": 0})

    y_preds = []
    errors = []
    maes, rmses, ndcgs = [], [], []

    borough_list = list(pd.unique(soos_df[split_var]))
    for i, borough in enumerate(borough_list):
        if verbose:
            print(f"Predicting borough {i+1}/{len(borough_list)}")

        train = soos_df[soos_df[split_var] != borough]  # leave out i'th district
        test = soos_df[soos_df[split_var] == borough]

        to_drop = ["venue_id", "latitude", "longitude", split_var, "_category"]
        train = train.drop(to_drop, axis=1)
        test = test.drop(to_drop, axis=1)

        # drop additional columns if given
        train = train.drop(additional_drops, axis=1)
        test = test.drop(additional_drops, axis=1)

        if train.isna().sum().sum() > 0 or test.isna().sum().sum() > 0:
            # impute with mean of train
            train = train.fillna(train.mean())
            test = test.fillna(train.mean())

        # remove unknown levels in test
        X_cv_cols = list(train.columns)
        for col in X_cv_cols:
            if "dummy_" in col and len(pd.unique(train[col])) <= 2:  # is binary column
                if np.sum(train[col]) == 0:  # no observations from this level
                    if verbose_drop:
                        print(f"removed column {col}, first occurence in test")
                    train.drop([col], axis=1, inplace=True)
                    test.drop([col], axis=1, inplace=True)
                    X_cv_cols.remove(col)

        X_train = train.drop(["total_visits"], axis=1)
        col_names = X_train.columns
        X_train = X_train.to_numpy()
        y_train = train["total_visits"].to_numpy()

        X_test = test.drop(["total_visits"], axis=1).to_numpy()
        y_test = test["total_visits"].to_numpy()

        if standardize:
            train_mean = X_train.mean()
            train_std = X_train.std()

            X_train = (X_train - train_mean) / train_std
            X_test = (X_test - train_mean) / train_std

        if "linear_model" in str(type(estimator)):
            estimator.fit(X=X_train, y=y_train)
        else:
            estimator.fit(X=X_train, y=y_train)

        y_pred_cv = estimator.predict(X_test)
        y_preds.extend(y_pred_cv)
        errors.extend([test - pred for test, pred in zip(y_test, y_pred_cv)])

        mae, rmse, ndcg = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        ndcgs.append(ndcg)

    error_df_soos["prediction"] = y_preds
    error_df_soos["error"] = errors

    all_sum = error_df_soos.shape[0]
    weights = [error_df_soos[error_df_soos[split_var] == bor].shape[0] / all_sum for bor in borough_list]

    avg_mae = sum(np.multiply(maes, weights))
    avg_rmse = sum(np.multiply(rmses, weights))
    avg_ndcg = sum(np.multiply(ndcgs, weights))

    mae_std = math.sqrt(np.average((maes-avg_mae)**2, weights=weights))
    rmse_std = math.sqrt(np.average((rmses-avg_rmse)**2, weights=weights))
    ndcg_std = math.sqrt(np.average((ndcgs-avg_ndcg)**2, weights=weights))

    if verbose:
        print("")
        print("Weighted metrics:")
        print(f"MAE:  {round(avg_mae, 2)} \u00B1 {round(mae_std, 2)}")
        print(f"RMSE: {round(avg_rmse, 2)} \u00B1 {round(rmse_std, 2)}")
        print(f"NDCG: {round(avg_ndcg, 3)} \u00B1 {round(ndcg_std, 3)}")

    if return_std:
        return error_df_soos, col_names, [maes, rmses, ndcgs], [mae_std, rmse_std, ndcg_std]
    else:
        return error_df_soos, col_names, [avg_mae, avg_rmse, avg_ndcg]

