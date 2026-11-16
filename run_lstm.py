import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 路径配置 --------------------------
TRAIN_PATH = r"D:\时序预测\train.csv"
OUT_DIR = r"D:\时序预测\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------- 数据加载与预处理 --------------------------
def load_train_data(csv_path):
    """仅加载train，包含count标签"""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)
    feats_cols = ["holiday", "workingday", "weather", "temp", "humidity", "windspeed"]
    df = df[["datetime", "count"] + feats_cols].copy()
    return df, feats_cols

df_raw, feat_cols = load_train_data(TRAIN_PATH)

# 时间切分：整体train文件内 80%训练，20% holdout验证
split_idx = int(len(df_raw) * 0.8)
df_tr = df_raw.iloc[:split_idx].copy()
df_hold = df_raw.iloc[split_idx:].copy()

# 标准化：只用训练集拟合scaler
scaler_feat = StandardScaler()
df_tr.loc[:, feat_cols] = scaler_feat.fit_transform(df_tr[feat_cols])
df_hold.loc[:, feat_cols] = scaler_feat.transform(df_hold[feat_cols])

# count做 log1p变换
df_tr["count_log"] = np.log1p(df_tr["count"])
df_hold["count_log"] = np.log1p(df_hold["count"])

# -------------------------- 构建样本：48h片段，前24h输入，后24h目标 --------------------------
def build_samples(df):
    samples = []
    arr_dt = df["datetime"].values
    arr_count_log = df["count_log"].values
    arr_feat = df[feat_cols].values.astype(np.float32)
    total = len(df)
    for i in range(total - 48 + 1):
        seg_dt = arr_dt[i:i+48]
        delta = seg_dt[1:] - seg_dt[:-1]
        # 校验每步间隔1小时
        if not all(x == np.timedelta64(3600, 's') for x in delta):
            continue
        src_x_feat = arr_feat[i:i+24]
        src_x_cnt = arr_count_log[i:i+24]
        tgt_y_cnt = arr_count_log[i+24:i+48]
        tgt_feat = arr_feat[i+24:i+48]
        samples.append({
            "src_feat": src_x_feat.astype(np.float32),
            "src_cnt": src_x_cnt.astype(np.float32),
            "tgt_feat": tgt_feat.astype(np.float32),
            "tgt_cnt": tgt_y_cnt.astype(np.float32),
        })
    return samples

train_samples = build_samples(df_tr)
hold_samples = build_samples(df_hold)
print(f"训练样本:{len(train_samples)}, holdout验证样本:{len(hold_samples)}")

class BikeDataset(data.Dataset):
    def __init__(self, sample_list):
        self.samples = sample_list
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        src_in = np.concatenate([s["src_feat"], s["src_cnt"][:,None]], axis=-1)
        return (
            torch.from_numpy(src_in),          # (24, n_feat+1) encoder输入
            torch.from_numpy(s["tgt_feat"]),   # (24, n_feat) decoder各步外生特征
            torch.from_numpy(s["tgt_cnt"])     # (24,) target count_log
        )

train_ds = BikeDataset(train_samples)
hold_ds = BikeDataset(hold_samples)

batch_size = 32
train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
hold_loader = data.DataLoader(hold_ds, batch_size=batch_size, shuffle=False)

# -------------------------- 模型：h1=32，h2=16，dropout=0.4 --------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, h1=32, h2=16, dropout=0.4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, h1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True, dropout=dropout)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        _, (h2, c2) = self.lstm2(out1)
        return h2[0,:,:], c2[0,:,:]

class DecoderCell(nn.Module):
    def __init__(self, feat_dim, h_dim=16, dropout=0.4):
        super().__init__()
        self.in_dim = feat_dim + 1
        self.lstm_cell = nn.LSTMCell(self.in_dim, h_dim)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(h_dim, 1)

    def forward(self, feat_step, prev_y, h_prev, c_prev):
        inp = torch.cat([feat_step, prev_y], dim=-1)
        inp = self.drop(inp)
        h_next, c_next = self.lstm_cell(inp, (h_prev, c_prev))
        pred = self.fc_out(h_next)
        return pred, h_next, c_next

class Seq2SeqModel(nn.Module):
    def __init__(self, input_enc_dim, feat_dec_dim, h1=32, h2=16, dropout=0.4):
        super().__init__()
        self.encoder = EncoderLSTM(input_enc_dim, h1, h2, dropout)
        self.decoder_cell = DecoderCell(feat_dec_dim, h2, dropout)

    def forward(self, src_in, tgt_feat, tgt_cnt, is_train=True):
        B = src_in.size(0)
        h, c = self.encoder(src_in)
        pred_seq = []
        if is_train:
            # 纯Teacher‑Forcing，训练全程喂真实标签
            prev_y = tgt_cnt[:, 0:1]
            for t in range(24):
                ft = tgt_feat[:, t, :]
                pred, h, c = self.decoder_cell(ft, prev_y, h, c)
                pred_seq.append(pred)
                if t < 23:
                    prev_y = tgt_cnt[:, t+1:t+2]
        else:
            # 推理：自回归，使用模型自己输出
            prev_y = torch.zeros((B,1), device=src_in.device)
            for t in range(24):
                ft = tgt_feat[:, t, :]
                pred, h, c = self.decoder_cell(ft, prev_y, h, c)
                pred_seq.append(pred)
                prev_y = pred
        out = torch.cat(pred_seq, dim=-1)
        return out

enc_input_dim = len(feat_cols) + 1
dec_feat_dim = len(feat_cols)
model = Seq2SeqModel(enc_input_dim, dec_feat_dim, h1=32, h2=16, dropout=0.4).to(device)

# -------------------------- 训练配置 【patience=6】 --------------------------
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
max_epoch = 150
patience = 6   # 减小早停容忍轮数
clip_norm = 1.0

best_hold_loss = float("inf")
no_improve_cnt = 0
best_state = None
train_loss_list = []
hold_loss_list = []

for epoch in range(max_epoch):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        src_in, tgt_feat, tgt_cnt = [x.to(device) for x in batch]
        pred = model(src_in, tgt_feat, tgt_cnt, is_train=True)
        loss = criterion(pred, tgt_cnt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_train_loss += loss.item() * src_in.size(0)
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_train_loss)

    model.eval()
    total_hold_loss = 0.0
    with torch.no_grad():
        for batch in hold_loader:
            src_in, tgt_feat, tgt_cnt = [x.to(device) for x in batch]
            pred = model(src_in, tgt_feat, tgt_cnt, is_train=False)
            loss = criterion(pred, tgt_cnt)
            total_hold_loss += loss.item() * src_in.size(0)
    avg_hold_loss = total_hold_loss / len(hold_loader.dataset)
    hold_loss_list.append(avg_hold_loss)

    print(f"Epoch {epoch+1:3d} | train_l1:{avg_train_loss:.5f} | hold_l1:{avg_hold_loss:.5f}")

    if avg_hold_loss < best_hold_loss:
        best_hold_loss = avg_hold_loss
        no_improve_cnt = 0
        best_state = model.state_dict()
    else:
        no_improve_cnt += 1
        if no_improve_cnt >= patience:
            print(f"Early stop epoch {epoch+1}, best hold loss:{best_hold_loss:.5f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(best_state, os.path.join(OUT_DIR, "model_tf_l1_p6.pt"))

# -------------------------- holdout集指标计算 --------------------------
def rmsle_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

model.eval()
y_true_all = []
y_pred_all = []
with torch.no_grad():
    for batch in hold_loader:
        src_in, tgt_feat, tgt_cnt_log = [x.to(device) for x in batch]
        pred_log = model(src_in, tgt_feat, tgt_cnt_log, is_train=False)
        pred_log_np = pred_log.cpu().numpy()
        tgt_log_np = tgt_cnt_log.cpu().numpy()
        pred_cnt = np.expm1(pred_log_np)
        true_cnt = np.expm1(tgt_log_np)
        y_true_all.append(true_cnt.reshape(-1))
        y_pred_all.append(pred_cnt.reshape(-1))

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
y_pred_all = np.clip(y_pred_all, 0, None)

rmsle_val = rmsle_np(y_true_all, y_pred_all)
rmse_val = np.sqrt(np.mean((y_true_all - y_pred_all)**2))
mae_val = np.mean(np.abs(y_true_all - y_pred_all))

print("\n==== Holdout验证集指标 ====")
print(f"RMSLE: {rmsle_val:.4f}")
print(f"RMSE : {rmse_val:.2f}")
print(f"MAE  : {mae_val:.2f}")

# loss绘图
plt.figure(figsize=(8,4))
plt.plot(train_loss_list, label="train L1(log‑count)")
plt.plot(hold_loss_list, label="holdout L1(log‑count)")
plt.xlabel("epoch")
plt.ylabel("L1 loss(log‑count)")
plt.legend()
plt.title("Loss curve | patience=6 | h1=32 h2=16 | L1Loss")
plt.savefig(os.path.join(OUT_DIR,"loss_tf_l1_p6.png"), dpi=150)
plt.close()
print(f"\n模型与图像输出目录: {OUT_DIR}")
