import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = r"D:\时序预测\train.csv"
TEST_PATH = r"D:\时序预测\test.csv"
OUTPUT_PATH = r"D:\时序预测\test_pred_xgb_baseline.csv"

# ===================== 加载数据，使用原始特征，不做周期编码 =====================
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# 原始基线特征（hour、month保留整数，无sin/cos）
FEATURE_COLS = [
    "year", "hour", "month", "holiday", "workingday",
    "weather", "temp", "humidity", "windspeed"
]
TARGET_COL = "count"

X_all = df_train[FEATURE_COLS].copy()
y_all_raw = df_train[TARGET_COL].copy()
y_all = np.log1p(y_all_raw)

# 严格时序切分：前80%用于调参/CV，后20% holdout仅做最终评估
split_idx = int(len(df_train) * 0.8)
X_80_train = X_all.iloc[:split_idx].copy()
y_80_train = y_all.iloc[:split_idx].copy()

X_holdout_val = X_all.iloc[split_idx:].copy()
y_holdout_val = y_all.iloc[split_idx:].copy()
y_holdout_val_raw = y_all_raw.iloc[split_idx:]

tscv = TimeSeriesSplit(n_splits=5)

# ===================== Optuna目标函数 =====================
def objective(trial: optuna.Trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_bin": 64,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        "n_jobs": -1,
        "random_state": 2024
    }
    rmsle_list = []
    for tr_idx, val_idx in tscv.split(X_80_train):
        X_tr, X_val = X_80_train.iloc[tr_idx], X_80_train.iloc[val_idx]
        y_tr, y_val = y_80_train.iloc[tr_idx], y_80_train.iloc[val_idx]
        y_val_raw = np.expm1(y_val)

        model = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=40, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred_log = model.predict(X_val)
        pred_real = np.clip(np.expm1(pred_log), 0, None)
        rmsle = np.sqrt(np.mean(np.square(np.log1p(pred_real) - np.log1p(y_val_raw))))
        rmsle_list.append(rmsle)
    return np.mean(rmsle_list)


print("===== 开始Optuna贝叶斯调参【基线，原始hour/month，无周期编码】 =====")
study = optuna.create_study(direction="minimize", study_name="xgb_baseline_bike")
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_bin": 64,
    "n_jobs": -1,
    "random_state": 2024
})

print(f"\n【最优CV平均RMSLE】: {study.best_value:.4f}")
print("【最优超参数】")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ===================== 使用最优参数，时序5折CV校验 =====================
print("\n===== 时序5折交叉验证（最优参数） =====")
cv_rmsle_list = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_80_train)):
    X_tr, X_val = X_80_train.iloc[tr_idx], X_80_train.iloc[val_idx]
    y_tr, y_val = y_80_train.iloc[tr_idx], y_80_train.iloc[val_idx]
    y_val_raw = np.expm1(y_val)

    model_cv = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50, **best_params)
    model_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred_log = model_cv.predict(X_val)
    pred_real = np.clip(np.expm1(pred_log), 0, None)
    rmsle = np.sqrt(np.mean(np.square(np.log1p(pred_real) - np.log1p(y_val_raw))))
    cv_rmsle_list.append(rmsle)
    print(f"fold {fold+1} RMSLE: {rmsle:.4f}")

print(f"5折平均RMSLE: {np.mean(cv_rmsle_list):.4f}, std: {np.std(cv_rmsle_list):.4f}")

# ===================== 训练最终模型，Holdout独立验证 =====================
print("\n===== 训练最终模型，Holdout独立验证集 =====")
model_final = xgb.XGBRegressor(
    n_estimators=3000,
    early_stopping_rounds=50,
    **best_params
)
model_final.fit(
    X_80_train, y_80_train,
    eval_set=[(X_80_train, y_80_train), (X_holdout_val, y_holdout_val)],
    verbose=False
)

pred_holdout_log = model_final.predict(X_holdout_val)
pred_holdout_real = np.clip(np.expm1(pred_holdout_log), 0, None)
y_ho_raw = y_holdout_val_raw.values

ho_rmsle = np.sqrt(np.mean(np.square(np.log1p(pred_holdout_real) - np.log1p(y_ho_raw))))
ho_rmse = np.sqrt(np.mean(np.square(pred_holdout_real - y_ho_raw)))
ho_mae = np.mean(np.abs(pred_holdout_real - y_ho_raw))

print(f"Holdout RMSLE: {ho_rmsle:.4f}")
print(f"Holdout RMSE : {ho_rmse:.2f}")
print(f"Holdout MAE  : {ho_mae:.2f}")
print(f"best_iteration: {model_final.best_iteration}")

# gain特征重要性
fi_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "gain_importance": model_final.feature_importances_
}).sort_values("gain_importance", ascending=False)
print("\n===== 特征重要性(gain) =====")
print(fi_df.round(4))

# ===================== test集预测输出 =====================
X_test = df_test[FEATURE_COLS].copy()
pred_test_log = model_final.predict(X_test)
pred_test_real = np.clip(np.expm1(pred_test_log), 0, None)

df_out = df_test.copy()
df_out["count_pred"] = pred_test_real.round(3)
df_out["count_pred_int"] = pred_test_real.round().astype(int)
df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf‑8‑sig")

print(f"\n预测结果保存至：{OUTPUT_PATH}")
print("说明：当前XGBoost版本不支持callback，不输出loss曲线图；本版本为基线，使用原始hour/month整数特征，无正余弦周期编码。")
