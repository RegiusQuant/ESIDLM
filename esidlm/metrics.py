from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calc_regression_metric(y_true, y_pred):
    return {
        "COUNT": len(y_true),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "PEARSON": pearsonr(y_true, y_pred)[0]
    }