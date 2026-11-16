# 共享单车需求预测 · XGBoost 实验记录

> 记录日期：2026-11-16
> 项目定位：简历项目（课程设计级），不做 Kaggle 正式提交
> 任务：小时级共享单车租赁量 `count` 预测

---

## 一、任务与数据

| 项目 | 内容 |
|---|---|
| 训练数据 | train.csv，10886 行，2011–2012 年小时级数据 |
| 测试数据 | test.csv，6493 行，无标签 |
| 目标 y | `count`，训练时用 `log1p(count)`，预测后 `expm1` 还原并截断 ≥0 |
| 最终特征 | `year, hour, month, holiday, workingday, weather, temp, humidity, windspeed` |
| 禁止使用 | `casual`、`registered`（不能拆开预测后再相加，会与 count 重复） |

### 数据事实（后续所有决策的依据）
- 训练集 count 范围 1–977，均值 191.6，标准差 181.1，原始偏度 1.24
- 无缺失值；`weather=4` 仅 1 条，`holiday=1` 仅 311 条（2.9%）
- 连续特征唯一值极少：temp 49 个、humidity 89 个、windspeed 28 个
- 文件为两年拼接：第 5422 行处月份从 12 回卷到 1
- 两年水平差异大：2011 均值 144.2 vs 2012 均值 238.6
- 原文件行序已是「年—月—日—小时」升序，但**没有 day 列**，因此任何 `sort_values` 都会打乱天序，只能保留原始行序

---

## 二、特征取舍决策（逐步确定）

| 特征 | 处理 | 理由 |
|---|---|---|
| `atemp` | **删除** | 与 temp 高度冗余 |
| `season` | **删除** | 用 month 替代，month 是更细粒度的季节表征 |
| `weekday` | **删除** | workingday / holiday 已覆盖工作日与节假日区分 |
| `hour` | **保留**（后期恢复） | 曾一度想删，但它是日内早晚高峰的核心信号，删掉会显著劣化 |
| `month` | 保留 | 捕捉月度/季节变化 |
| `year` | 新增 | 承载 2011→2012 的整体量级抬升，没有它模型无法解释年度漂移 |
| `casual/registered` | 不用 | 方案讨论过拆分预测再求和，但工作量翻倍、误差叠加，未采纳 |

### 明确不做的处理
- 不做 PCA（0-1 与多分类离散特征不满足 PCA 的连续线性假设，且破坏可解释性）
- 不做 VIF / 相关系数共线性检验（数据侧要求只删指定列，不做分析）
- 不构造 lag / rolling 等手工时序特征（XGB 只用单时刻静态外生特征，作为与深度学习的对照基线）

---

## 三、切分与验证方案（无时间穿越）

1. **Holdout**：按原始时间顺序，前 80%（8708 行，至 2012-08）训练，后 20%（2178 行，2012-08 起）验证。
   - holdout 全程不参与 CV、不参与调参，只做最终评估。
2. **TimeSeriesSplit(5)**：仅在 80% 训练集内部执行，每折训练集时间严格早于验证集。
3. 禁止 `shuffle=True`，禁止任何打乱时间顺序的操作。

> 说明：曾一度考虑「加入 lag 特征后随机切分也算无泄露」，但本方案无 lag，样本独立，切分方式只影响分布覆盖，因此仍按用户要求坚持严格时序切分。

### n_estimators 的确定方式（踩过的坑）
- 时序 5 折早期折训练样本少（仅 1453 行），会过早早停，best_iter 系统性偏低（152–349）
- 直接取 CV 平均 best_iter 重训全量会**欠拟合**
- 实测：246 轮 → 0.3252；400 轮 → 0.3218；600 轮 → 0.3216
- 最终改为**训练集内部末尾 10% 早停**（内部 9:1 切分）确定轮数，全程不外泄

---

## 四、XGBoost 参数策略

### 固定结构参数
| 参数 | 取值 | 依据 |
|---|---|---|
| objective | reg:squarederror | log1p 目标下该损失域的 rmse 即 RMSLE |
| eval_metric | rmse | 与 RMSLE 对齐 |
| tree_method | hist | 分桶近似算法，小数据最快 |
| max_bin | 64 | temp 仅 49 个唯一值；实测 16 桶劣化到 0.4425，256 桶无提升 |
| grow_policy | depthwise | 数据量小，leaf-wise 更易过拟合 |

### 手工基准配置（未调参时）
`lr=0.05, max_depth=5, min_child_weight=1, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0, gamma=0`
分层 5 折实测 RMSLE 0.3640；单参数消融结论：除 `max_bin<32`、`max_depth≤3` 会明显退化外，其余参数在 0.364–0.369 摆动，**调参边际收益 ≤0.003**。

---

## 五、Optuna 贝叶斯调参

- 迭代 30 次，目标函数 = 80% 训练集内部 TimeSeriesSplit 5 折平均 RMSLE
- holdout 完全隔离，不参与搜索
- 搜索空间：`lr[0.02,0.1]`、`max_depth[3,7]`、`min_child_weight[1,12]`、`subsample[0.6,0.9]`、`colsample_bytree[0.6,0.95]`、`reg_lambda[0.5,8]`、`reg_alpha[0,1]`、`gamma[0,1.5]`

### 最终最优参数（基线版本）
```
learning_rate      0.1
max_depth          3
min_child_weight   3
subsample          0.6766
colsample_bytree   0.6672
reg_lambda         5.6060
reg_alpha          0.1709
gamma              0.0103
objective          reg:squarederror
eval_metric        rmse
tree_method        hist
max_bin            64
random_state       2024
```
> 解读：最优树深度只有 3、L2 正则 5.6 偏大，说明数据噪声与季节漂移明显，模型需要强正则、低容量，否则会记住局部季节噪声。

---

## 六、实验结果

### 6.1 基线版本（原始 hour/month 整数特征）★ 采用
| 折数 | RMSLE |
|---|---|
| fold1 | 0.4928 |
| fold2 | 0.3335 |
| fold3 | 0.5227 |
| fold4 | 0.4045 |
| fold5 | 0.3030 |

- 5 折平均：**0.4113**，标准差 **0.0859**
- 最优参数下 best_iteration = 521

| Holdout 指标 | 数值 |
|---|---|
| RMSLE | **0.3189** |
| RMSE | 60.40 |
| MAE | 38.76 |

**特征重要性（gain）**
| feature | gain |
|---|---|
| hour | 0.5114 |
| year | 0.1320 |
| workingday | 0.1143 |
| temp | 0.0926 |
| month | 0.0586 |
| humidity | 0.0416 |
| weather | 0.0337 |
| holiday | 0.0087 |
| windspeed | 0.0072 |

> hour 独占一半增益，是绝对主特征；year 第二，承载年度量级差；holiday 与 windspeed 几乎无区分力。

### 6.2 消融版本（正余弦周期编码，已放弃）
特征换成 `hour_sin/hour_cos/month_sin/month_cos`，删除原始 hour/month。

| 指标 | 数值 |
|---|---|
| 最优 CV RMSLE（Optuna best） | 0.5221 → 该参数下复跑 0.5198 |
| 5 折均值 / 标准差 | 0.5198 / **0.1378** |
| Holdout RMSLE | 0.3745 |
| Holdout RMSE / MAE | 80.64 / 53.60 |
| best_iteration | 425 |

特征重要性：`hour_sin 0.3314 + hour_cos 0.2900` 合计约 0.62；`month_sin 0.0349`、`month_cos 0.0190` 极低。

> 结论：周期编码没缓解季节漂移，**折间标准差从 0.0859 涨到 0.1378**，holdout 也从 0.3189 退化到 0.3745。树模型靠分裂切分，对连续三角函数特征利用能力差，整数 hour/month 反而更好切。该套周期编码保留给 LSTM / Transformer 使用。

### 6.3 早期对照数字（供参考）
| 场景 | RMSLE |
|---|---|
| 常数预测（训练集 log 均值）在全局 80/20 | 1.4409 |
| 常数预测在年内切分 | 1.3363 / 1.3550 |
| 无 year 特征的 XGB，month×hour 分层 5 折 | 0.3673 |
| 无 year 特征的 XGB，全局时序 80/20 | 0.4933 |
| 无 year 特征的 XGB，跨年 holdout | 0.5918 |

---

## 七、结论

1. **XGB 基线采用原始整数 hour/month**，加 year 特征后 holdout RMSLE 0.3189，优于周期编码版本的 0.3745。
2. 数据集存在明显季节与年度漂移，时序 5 折折间标准差 0.086；**不能只报 holdout，必须同时报 CV 均值 + 标准差**。
3. 本模型只用单时刻静态外生特征，**无历史时序信息**，作为 LSTM / GRU / Transformer 的对照基线。
4. 调参空间接近饱和：手工配置 0.3640 vs Optuna 最优 0.4113（CV 口径不同，前者是分层折、后者是时序折），真正的瓶颈在特征与切分方式，不在参数。

---

## 八、环境踩坑记录（后续复用）

| 报错 | 原因 | 解决 |
|---|---|---|
| `partially initialized module 'xgboost'` | 脚本文件命名为 `xgboost.py`，循环导入 | 改名，禁止用库名做文件名 |
| `XGBModel.fit() got an unexpected keyword argument 'eval_metric'` | eval_metric 是构造参数，不是 fit 参数 | 放到 `XGBRegressor(...)` 里 |
| `... 'eval_result'` | 旧版 XGB 的 fit 不支持 eval_result | 不采集 loss 历史 |
| `cannot import name 'EvaluationCallback'` | 版本太旧无此回调 | 手写 TrainingCallback 或不画曲线 |
| `... 'callbacks'` | 旧版 fit 不支持 callbacks | 放弃 loss 曲线作图 |

---

## 九、输出文件

| 文件 | 说明 |
|---|---|
| `D:\时序预测\test_pred_xgb_baseline.csv` | 最终基线预测，含 `count_pred`（3 位小数）、`count_pred_int`（整数） |
| `run_xgb_baseline.py` | 基线完整代码（Optuna + 时序切分） |

---

## 十、下一步

- 搭 LSTM / GRU / Encoder-only Transformer，**完全复用本记录的切分方式、指标口径、特征集合**
- 深度学习可使用正余弦周期编码（XGB 上无效，但序列模型上预期有效）
- 保持对照公平：同一 holdout、同一 RMSLE 计算方式、同一梯度式报告 CV 均值 + 标准差

> （注：部分内容由豆包 AI 生成）