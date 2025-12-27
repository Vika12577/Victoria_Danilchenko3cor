"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import numpy as np
seed = 322
folds = 5
eps = 1e-6
np.random.seed(seed)
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    submission = pd.DataFrame({
        "row_id": predictions["row_id"],
        "price_p05": predictions["price_p05"],
        "price_p95": predictions["price_p95"]
    })
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path
    
    
def dop_feat(tabl):
    tabl = tabl.copy()
    tabl["shoplog"] = np.log1p(tabl["n_stores"])
    tabl["rainlog"]   = np.log1p(tabl["precpt"])
    tabl["dsin"] = np.sin(2 * np.pi * tabl["dow"] / 7)
    tabl["dcos"] = np.cos(2 * np.pi * tabl["dow"] / 7)
    tabl["msin"] = np.sin(2 * np.pi * tabl["month"] / 12)
    tabl["mcos"] = np.cos(2 * np.pi * tabl["month"] / 12)
    tabl["daysin"] = np.sin(2 * np.pi * tabl["day_of_month"] / 31)
    tabl["daycos"] = np.cos(2 * np.pi * tabl["day_of_month"] / 31)
    tabl["wsin"] = np.sin(2 * np.pi * tabl["week_of_year"] / 52)
    tabl["wcos"] = np.cos(2 * np.pi * tabl["week_of_year"] / 52)
    tabl["mstart"]  = (tabl["day_of_month"] <= 10).astype(int)
    tabl["mmiddle"] = ((tabl["day_of_month"] > 10) & (tabl["day_of_month"] <= 20)).astype(int)
    tabl["mend"]    = (tabl["day_of_month"] > 20).astype(int)

    return tabl

def func_iou(low_true, up_true, low_pred, up_pred):
    min_up = np.minimum(up_true, up_pred)
    max_low = np.maximum(low_true, low_pred)
    inter = np.maximum(0, min_up - max_low)
    union = (up_true - low_true) + (up_pred - low_pred) - inter
    return np.mean(inter / np.maximum(union, eps))


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    train = pd.read_csv("data/train.csv", parse_dates=["dt"])
    test  = pd.read_csv("data/test.csv",  parse_dates=["dt"])

    train["n_stores"] = train["n_stores"].clip(lower=0)
    test["n_stores"]  = test["n_stores"].clip(lower=0)

    train["precpt"] = train["precpt"].clip(lower=0)
    test["precpt"]  = test["precpt"].clip(lower=0)

    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed)

    mask = iso.fit_predict(train[["price_p05", "price_p95", "n_stores"]])
    train = train[mask == 1].reset_index(drop=True)

    train = dop_feat(train)
    test  = dop_feat(test)
    train["price_center"] = (train["price_p05"] + train["price_p95"]) / 2
    train["price_width"]  = train["price_p95"] - train["price_p05"]

    stats = (train.groupby("product_id")[["price_center", "price_width"]].mean())
    scaler = StandardScaler()
    Xc = scaler.fit_transform(np.log1p(stats))
    kmeans = KMeans(n_clusters=30, n_init=10, random_state=seed)
    stats["clust"] = kmeans.fit_predict(Xc)

    train = train.merge(stats["clust"], on="product_id", how="left")
    test  = test.merge(stats["clust"], on="product_id", how="left")
    train = train.sort_values(["product_id", "dt"])
    train["prev"] = train.groupby("product_id")["price_center"].shift(1)
    train["smooth"] = (train.groupby("product_id")["price_center"].transform(lambda x: x.rolling(3, min_periods=1).mean()))

    median_center = train["price_center"].median()
    test["prev"] = median_center
    test["smooth"] = median_center


    cols = ["shoplog", "holiday_flag", "activity_flag",
            "rainlog", "avg_temperature", "avg_humidity",
            "avg_wind_level", "dsin", "dcos","msin", "mcos",
            "daysin", "daycos","wsin", "wcos","mstart",
            "mmiddle", "mend","clust","management_group_id",
            "first_category_id","second_category_id",
            "third_category_id","prev", "smooth"]

    cat_cols = ["clust","management_group_id","first_category_id",
        "second_category_id","third_category_id"]
        
    def model_regr(X, y, alpha):
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=seed,
            verbose=-1)
        model.fit(X, y, categorical_feature=cat_cols)
        return model

    gkf = GroupKFold(n_splits=folds)
    vals = []

    for f, (tr_i, va_i) in enumerate(gkf.split(train, groups=train["product_id"])):
        tr = train.iloc[tr_i]
        va = train.iloc[va_i]

        bot = model_regr(tr[cols], tr["price_p05"], 0.06)
        top = model_regr(tr[cols], tr["price_p95"], 0.94)
        pbot = bot.predict(va[cols])
        ptop = top.predict(va[cols])

        width = ptop - pbot
        cw = tr.groupby("clust")["price_width"].mean().to_dict()
        scale = va["clust"].map(cw).fillna(width.mean())
        bias = 0.012 * np.tanh(scale / scale.mean())

        pbot *= (1 - bias)
        ptop *= (1 + bias)
        ptop = np.maximum(ptop, pbot + eps)

        score = func_iou(va["price_p05"].values,va["price_p95"].values, pbot, ptop)

        vals.append(score)
        print(f"Fold {f+1}: IoU = {score:.5f}")

    print("CV mean IoU:", np.mean(vals))

    X_train = train[cols]
    X_test  = test[cols]

    bot_final = model_regr(X_train, train["price_p05"], 0.06)
    top_final = model_regr(X_train, train["price_p95"], 0.94)

    test["price_p05"] = bot_final.predict(X_test)
    test["price_p95"] = top_final.predict(X_test)

    width = test["price_p95"] - test["price_p05"]
    center = (test["price_p05"] + test["price_p95"]) / 2
    max_w = width.quantile(0.98)
    width = np.minimum(width, max_w)

    test["price_p05"] = center - width / 2
    test["price_p95"] = center + width / 2
    test["price_p95"] = np.maximum(test["price_p95"], test["price_p05"] + eps)

    create_submission(test)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
