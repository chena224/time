# 共享单车小时级时序预测
Kaggle Bike Sharing Demand 时序预测项目
任务：基于连续24小时历史时序数据，预测未来24小时逐小时共享单车租赁数量。

## 项目介绍
本项目对比传统机器学习XGBoost与深度时序模型Encoder-Decoder LSTM。
- 模型1：XGBoost，多输出回归，加入周期特征作为基线模型
- 模型2：Encoder-Decoder LSTM（LSTM+LSTMCell自回归），Teacher-Forcing训练，L1Loss，早停+全套正则

时序任务严格遵循时间顺序切分训练/验证集，避免未来信息泄露。
数据集来源：Kaggle Bike Sharing Demand

## 项目文件说明
├── outputs/                     # 模型权重、loss 曲线图片、预测输出
├── lstm_car.md                  # LSTM 模型完整实验报告
├── XGBoost 实验记录.md           # XGBoost 基线实验记录
├── run_lstm.py                  # Encoder-Decoder LSTM 训练脚本
├── run_xgboost.py               # XGBoost 基线训练脚本
├── train.csv/test.csv         # Kaggle 原始数据集
├── test_pred_xgb_baseline.csv   # XGB 基线预测结果
├── test_pred_xgb_cyclical.csv   # 增加周期特征 XGB 预测结果
├── requirements.txt             # Python 依赖
└── .gitignore

## 环境安装
```bash
pip install -r requirements.txt

# 训练LSTM Seq2Seq模型
python run_lstm.py

# 训练XGBoost基线模型
python run_xgboost.py

## 验结果摘要

### LSTM Encoder-Decoder

Holdout 验证集最优指标：

- RMSLE: 1.3271
- RMSE : 261.50
- MAE  : 188.19

> 
> 训练 Epoch15 达到验证集最优，后期出现明显过拟合，采用早停策略保存最优权重。

### XGBoost 基线

详见`XGBoost实验记录.md`

## 技术栈

Python, Pandas, NumPy, Scikit-learn, PyTorch, XGBoost, Matplotlib



1. 时序数据集**时间顺序切分训练 / 验证集**，杜绝数据泄露，符合时序预测工程规范
2. 搭建 Encoder-Decoder LSTM 自回归时序预测框架，Teacher-Forcing 训练
3. 组合正则策略：L1 鲁棒损失、Dropout、AdamW 权重衰减、梯度裁剪、早停抑制小数据集过拟合
4. 基线对比：XGBoost 多输出回归，验证深度学习模型## 验结果摘要

### LSTM Encoder-Decoder

Holdout 验证集最优指标：

- RMSLE: 1.3271
- RMSE : 261.50
- MAE  : 188.19

> 
> 训练 Epoch15 达到验证集最优，后期出现明显过拟合，采用早停策略保存最优权重。

### XGBoost 基线

详见`XGBoost实验记录.md`

## 技术栈

Python, Pandas, NumPy, Scikit-learn, PyTorch, XGBoost, Matplotlib

## 亮点（简历可直接摘抄）

1. 时序数据集**时间顺序切分训练 / 验证集**，杜绝数据泄露，符合时序预测工程规范
2. 搭建 Encoder-Decoder LSTM 自回归时序预测框架，Teacher-Forcing 训练
3. 组合正则策略：L1 鲁棒损失、Dropout、AdamW 权重衰减、梯度裁剪、早停抑制小数据集过拟合
4. 基线对比：XGBoost 多输出回归，验证深度学习模型

## 改进方向

- 输入高斯噪声数据增强
- 增加星期、月份等周期性特征
- 尝试时序 Transformer 模型
- 面向 RMSLE 定制损失函数


