# -*- coding: utf-8 -*-
"""
AI时序预测模型训练模块
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统

输入特征:
  课表使用强度指数、室外温度、室外湿度、室内温度、
  室内湿度、CO₂浓度、历史能耗(过去24小时)
预测目标: 未来1小时空调能耗
模型:
  首选 LSTM (PyTorch)，若 torch 不可用则回退 RandomForest
  ——确保代码在任何环境下均可运行
划分: 训练/验证/测试 = 7:1.5:1.5
输出:
  models/energy_model.pkl          (模型文件)
  results/prediction_vs_actual.png (预测对比图)
  results/model_metrics.txt        (精度指标)
=====================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============ 中文字体适配（跨平台：Windows / macOS / Linux / Streamlit Cloud） ============
from matplotlib import font_manager as _fm
_cjk_candidates = ['Noto Sans CJK SC','Noto Sans SC','Noto Sans CJK JP',
    'WenQuanYi Micro Hei','WenQuanYi Zen Hei','Source Han Sans CN',
    'AR PL UMing CN','Droid Sans Fallback','PingFang SC','Heiti SC',
    'Arial Unicode MS','SimHei','Microsoft YaHei','DejaVu Sans']
_avail = {f.name for f in _fm.fontManager.ttflist}
for _n in _cjk_candidates:
    if _n in _avail:
        plt.rcParams['font.sans-serif'] = [_n, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False
# =====================================================================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    print("检测到 PyTorch，使用 LSTM 模型")
except ImportError:
    HAS_TORCH = False
    print("未检测到 PyTorch，回退使用 RandomForest 模型")

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 模型超参数
LOOKBACK = 24         # 过去24小时作为输入序列
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# 特征列
FEATURE_COLS = [
    'course_intensity', 'outdoor_temp', 'outdoor_humidity',
    'indoor_temp', 'indoor_humidity', 'co2', 'ac_energy'
]


# ============================================================
# 一、数据预处理
# ============================================================
def load_and_preprocess(train_ratio=0.7, val_ratio=0.15):
    """加载数据并预处理。

    两项修复：
    1. 数据泄漏——scaler 仅用【训练集】拟合，再对全量数据 transform；
    2. 跨季节测试集——测试集按【每周日】抽样，覆盖全年各季节，
       避免按时间顺序切分导致测试集固定落在年末秋冬、指标虚高。
    """
    csv_path = os.path.join(DATA_DIR, 'classroom_energy_dataset.csv')
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 添加时间特征
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    feature_cols_all = FEATURE_COLS + ['hour', 'day_of_week', 'month', 'is_weekend']

    # 测试集：每周日（dayofweek==6），约占 1/7 ≈ 14%，全年均匀分布
    test_mask = (df['datetime'].dt.dayofweek == 6).values

    # 非测试数据按时间顺序切分训练/验证（比例 train_ratio : val_ratio）
    non_test_idx = np.where(~test_mask)[0]
    train_cut = int(len(non_test_idx) * train_ratio / (train_ratio + val_ratio))
    train_idx = non_test_idx[:train_cut]
    val_idx = non_test_idx[train_cut:]

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    # scaler 只用训练集拟合，避免验证/测试集信息泄漏
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[train_idx][feature_cols_all])
    df_scaled = df.copy()
    df_scaled[feature_cols_all] = scaler.transform(df[feature_cols_all])

    # 保存scaler用于后续推理
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(feature_cols_all, os.path.join(MODEL_DIR, 'feature_cols.pkl'))

    return df, df_scaled, feature_cols_all, scaler, train_mask, val_mask, test_mask


def create_sequences(df_scaled, feature_cols, lookback=24):
    """
    创建时序序列:
      X[i] = 特征矩阵 (lookback, n_features)
      y[i] = ac_energy 在 i 时刻
    """
    data = df_scaled[feature_cols].values
    target = df_scaled['ac_energy'].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def split_data(X, y, lookback, train_mask, val_mask, test_mask):
    """按掩码划分 训练/验证/测试。

    train_mask/val_mask/test_mask 为【原始数据】的布尔掩码（长度 = 原始行数）。
    序列样本 i 的目标对应原始数据第 i+lookback 行，故掩码需去掉前 lookback 行。
    """
    sample_train = train_mask[lookback:]
    sample_val = val_mask[lookback:]
    sample_test = test_mask[lookback:]

    X_train, y_train = X[sample_train], y[sample_train]
    X_val, y_val = X[sample_val], y[sample_val]
    X_test, y_test = X[sample_test], y[sample_test]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# 二、LSTM 模型定义 (PyTorch) —— 仅在 torch 可用时定义
# ============================================================
if HAS_TORCH:
    class LSTMModel(nn.Module):
        """LSTM 时序预测模型"""
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            out, (hn, cn) = self.lstm(x)
            out = out[:, -1, :]  # 取最后时刻
            out = self.fc(out)
            return out.squeeze()


def train_lstm(X_train, y_train, X_val, y_val, feature_cols):
    """训练 LSTM 模型"""
    input_size = len(feature_cols)

    # 转 tensor
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    print(f"\n开始训练 LSTM (epochs={EPOCHS}, lookback={LOOKBACK})...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model.pth'))

    # 绘制训练曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM 训练损失曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'training_loss.png'), dpi=150)
    plt.close()

    return model


def predict_lstm(model, X):
    """LSTM 推理"""
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).numpy()
    return preds


# ============================================================
# 三、RandomForest 回退方案
# ============================================================
def create_lag_features(df, target_col='ac_energy', lags=24):
    """为 RandomForest 创建滞后特征"""
    df_feat = df.copy()
    for lag in range(1, lags + 1):
        df_feat[f'energy_lag_{lag}'] = df_feat[target_col].shift(lag)
    df_feat = df_feat.dropna()
    return df_feat


def train_rf(df_scaled, feature_cols, train_mask, val_mask, test_mask):
    """训练 RandomForest 模型"""
    # 创建滞后特征
    df_lag = create_lag_features(df_scaled, 'ac_energy', LOOKBACK)
    lag_cols = [f'energy_lag_{l}' for l in range(1, LOOKBACK + 1)]
    # 排除 ac_energy 本身（目标变量），只用滞后值作为历史能耗特征
    rf_features = [f for f in feature_cols if f != 'ac_energy']
    all_features = rf_features + lag_cols

    X = df_lag[all_features].values
    y = df_lag['ac_energy'].values

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X, y, LOOKBACK, train_mask, val_mask, test_mask)

    print("\n训练 RandomForest 模型...")
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    print("  训练完成")

    # 保存
    joblib.dump({
        'model': model,
        'feature_cols': all_features,
        'lookback': LOOKBACK,
    }, os.path.join(MODEL_DIR, 'rf_model.pkl'))

    return model, all_features, (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# 四、评估指标
# ============================================================
def calculate_metrics(y_true, y_pred):
    """计算 MAPE, RMSE, R²"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # MAPE (避免除零)
    mask = y_true > 0.001
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {'MAPE': mape, 'RMSE': rmse, 'R2': r2}


def evaluate_by_season(y_true, y_pred, months):
    """按季节分组评估模型表现。

    months: 与 y_true/y_pred 一一对应的月份数组（1-12）。
    返回 {季节: {'MAPE','RMSE','R2','样本数'}}，用于揭示不同季节的预测难度差异，
    避免被"空调多关闭的秋冬"拉高的整体指标掩盖夏季真实误差。
    """
    season_map = {3: '春季', 4: '春季', 5: '春季',
                  6: '夏季', 7: '夏季', 8: '夏季',
                  9: '秋季', 10: '秋季', 11: '秋季',
                  12: '冬季', 1: '冬季', 2: '冬季'}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    seasons = np.array([season_map.get(int(m), '未知') for m in months])

    result = {}
    for s in ['春季', '夏季', '秋季', '冬季']:
        mask = seasons == s
        if mask.sum() == 0:
            continue
        m_ = calculate_metrics(y_true[mask], y_pred[mask])
        result[s] = {**m_, '样本数': int(mask.sum())}
    return result


def plot_predictions(y_true, y_pred, title, save_path):
    """绘制预测对比图"""
    n = len(y_true)
    plt.figure(figsize=(14, 5))
    plt.plot(range(n), y_true, label='真实值', color='#2196F3', linewidth=1.2)
    plt.plot(range(n), y_pred, label='预测值', color='#4CAF50', linewidth=1.2, alpha=0.85)
    plt.xlabel('时间（小时）')
    plt.ylabel('空调能耗 (归一化)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 五、主函数
# ============================================================
def main():
    print("=" * 60)
    print("  AI时序预测模型训练")
    print("=" * 60)

    # 加载数据（scaler 仅用训练集拟合；测试集=每周日抽样，覆盖全年各季节）
    df, df_scaled, feature_cols, scaler, train_mask, val_mask, test_mask = load_and_preprocess()
    print(f"数据集: {len(df)} 行, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"特征列: {feature_cols}")
    print(f"样本划分: 训练 {int(train_mask.sum())} | 验证 {int(val_mask.sum())} | "
          f"测试 {int(test_mask.sum())}（测试集=每周日，覆盖全年各季节）")

    # 与序列样本一一对应的月份标签（用于按季节评估）
    sample_months = df['month'].values[LOOKBACK:]
    sample_test_mask = test_mask[LOOKBACK:]

    if HAS_TORCH:
        # ===== LSTM 方案 =====
        X, y = create_sequences(df_scaled, feature_cols, LOOKBACK)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            X, y, LOOKBACK, train_mask, val_mask, test_mask)
        print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

        model = train_lstm(X_train, y_train, X_val, y_val, feature_cols)

        # 验证集预测
        val_pred = predict_lstm(model, X_val)
        val_metrics = calculate_metrics(y_val, val_pred)

        # 测试集预测
        y_pred = predict_lstm(model, X_test)
        metrics = calculate_metrics(y_test, y_pred)
        plot_predictions(y_test, y_pred, 'LSTM 能耗预测：真实值 vs 预测值',
                         os.path.join(RESULT_DIR, 'prediction_vs_actual.png'))

        model_type = 'lstm'
        model_feature_cols = feature_cols
    else:
        # ===== RandomForest 回退方案 =====
        model, all_features, (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            train_rf(df_scaled, feature_cols, train_mask, val_mask, test_mask)
        print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

        # 验证集
        val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, val_pred)

        # 测试集
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        plot_predictions(y_test, y_pred, 'RandomForest 能耗预测：真实值 vs 预测值',
                         os.path.join(RESULT_DIR, 'prediction_vs_actual.png'))

        model_type = 'rf'
        model_feature_cols = all_features

    # 保存模型类型标记（两个分支共用）
    joblib.dump({'model_type': model_type, 'feature_cols': model_feature_cols,
                 'lookback': LOOKBACK},
                os.path.join(MODEL_DIR, 'model_info.pkl'))
    joblib.dump(model_type, os.path.join(MODEL_DIR, 'energy_model.pkl'))

    # 按季节评估测试集，暴露不同季节的预测难度差异
    season_metrics = evaluate_by_season(y_test, y_pred, sample_months[sample_test_mask])

    # 综合精度（验证集+测试集平均，更具代表性）
    avg_mape = (val_metrics['MAPE'] + metrics['MAPE']) / 2
    avg_rmse = (val_metrics['RMSE'] + metrics['RMSE']) / 2
    avg_r2 = (val_metrics['R2'] + metrics['R2']) / 2

    # 输出精度指标
    print("\n" + "=" * 50)
    print("  模型精度指标（MAPE/RMSE 基于归一化目标值，R² 为无量纲）")
    print("=" * 50)
    print(f"{'指标':<10} {'验证集':>12} {'测试集':>12} {'综合(均值)':>12}")
    print("-" * 50)
    print(f"{'MAPE':<10} {val_metrics['MAPE']:>11.2f}% {metrics['MAPE']:>11.2f}% {avg_mape:>11.2f}%")
    print(f"{'RMSE':<10} {val_metrics['RMSE']:>12.4f} {metrics['RMSE']:>12.4f} {avg_rmse:>12.4f}")
    print(f"{'R²':<10} {val_metrics['R2']:>12.4f} {metrics['R2']:>12.4f} {avg_r2:>12.4f}")
    print("=" * 50)

    # 按季节评估打印
    print("\n  按季节评估（测试集）")
    print("-" * 50)
    print(f"{'季节':<8} {'样本数':>8} {'MAPE':>10} {'RMSE':>10} {'R²':>10}")
    for s in ['春季', '夏季', '秋季', '冬季']:
        if s in season_metrics:
            m_ = season_metrics[s]
            print(f"{s:<8} {m_['样本数']:>8} {m_['MAPE']:>9.2f}% {m_['RMSE']:>10.4f} {m_['R2']:>10.4f}")
    print("-" * 50)

    # 保存指标到文件
    with open(os.path.join(RESULT_DIR, 'model_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("模型精度指标（MAPE/RMSE 基于归一化目标值，R² 为无量纲）\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型类型: {'LSTM (PyTorch)' if HAS_TORCH else 'RandomForest (scikit-learn)'}\n")
        f.write(f"训练/验证/测试划分: 约 7:1.5:1.5（测试集=每周日抽样覆盖全年；scaler 仅用训练集拟合，无数据泄漏）\n")
        f.write(f"输入序列长度(lookback): {LOOKBACK} 小时\n")
        f.write(f"特征列: {feature_cols}\n\n")
        f.write(f"{'指标':<10} {'验证集':>12} {'测试集':>12} {'综合(均值)':>12}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'MAPE':<10} {val_metrics['MAPE']:>11.2f}% {metrics['MAPE']:>11.2f}% {avg_mape:>11.2f}%\n")
        f.write(f"{'RMSE':<10} {val_metrics['RMSE']:>12.4f} {metrics['RMSE']:>12.4f} {avg_rmse:>12.4f}\n")
        f.write(f"{'R²':<10} {val_metrics['R2']:>12.4f} {metrics['R2']:>12.4f} {avg_r2:>12.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("按季节评估（测试集）\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'季节':<8} {'样本数':>8} {'MAPE':>10} {'RMSE':>10} {'R²':>10}\n")
        for s in ['春季', '夏季', '秋季', '冬季']:
            if s in season_metrics:
                m_ = season_metrics[s]
                f.write(f"{s:<8} {m_['样本数']:>8} {m_['MAPE']:>9.2f}% {m_['RMSE']:>10.4f} {m_['R2']:>10.4f}\n")
        f.write("\n注：综合指标为验证集与测试集的均值，更具代表性。\n")
        f.write("    MAPE/RMSE 基于 MinMaxScaler 归一化后的目标值计算；R² 为无量纲，不受归一化影响。\n")
        f.write("    测试集为每周日抽样，覆盖全年各季节，故分季节指标能如实反映全年表现。\n")
        f.write("    冬季/过渡季因大量时段能耗为0，MAPE 仅基于非零样本，可能波动较大。\n")

    print(f"\n预测对比图已保存: results/prediction_vs_actual.png")
    if HAS_TORCH:
        print(f"训练损失曲线已保存: results/training_loss.png")
    print(f"精度指标已保存: results/model_metrics.txt")
    print(f"模型已保存: models/")
    print("\n模型训练完成！")


def load_inference_artifacts():
    """加载训练好的模型、scaler、特征列，供推理使用。

    返回 (rf_data, scaler, feature_cols_all)；若模型文件缺失返回 None。
    """
    rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
    cols_path = os.path.join(MODEL_DIR, 'feature_cols.pkl')
    if not all(os.path.exists(p) for p in [rf_path, scaler_path, cols_path]):
        return None
    rf_data = joblib.load(rf_path)              # {'model', 'feature_cols', 'lookback'}
    scaler = joblib.load(scaler_path)           # MinMaxScaler（在 feature_cols_all 上 fit）
    feature_cols_all = joblib.load(cols_path)   # 归一化用到的全部特征列
    return rf_data, scaler, feature_cols_all


def predict_ac_energy(df):
    """用训练好的 RandomForest 对 df 逐行预测 ac_energy（真实尺度）。

    复用与训练完全相同的预处理（scaler.transform + create_lag_features），
    保证特征对齐，避免手动构造特征时出错。

    参数:
      df: 需含 datetime 及 course_intensity, outdoor_temp, outdoor_humidity,
          indoor_temp, indoor_humidity, co2, ac_energy 列。
    返回:
      pd.Series 预测值（索引与 df 对齐，前 lookback 行无足够历史，未返回）。
    """
    artifacts = load_inference_artifacts()
    if artifacts is None:
        return None
    rf_data, scaler, feature_cols_all = artifacts

    model = rf_data['model']
    all_features = rf_data['feature_cols']
    lookback = rf_data['lookback']

    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['hour'] = d['datetime'].dt.hour
    d['day_of_week'] = d['datetime'].dt.dayofweek
    d['month'] = d['datetime'].dt.month
    d['is_weekend'] = (d['day_of_week'] >= 5).astype(int)

    # 归一化（与训练一致：scaler 在 feature_cols_all 上 fit）
    d_scaled = d.copy()
    d_scaled[feature_cols_all] = scaler.transform(d[feature_cols_all])

    # 滞后特征（与训练一致）
    d_lag = create_lag_features(d_scaled, 'ac_energy', lookback)
    X = d_lag[all_features].values

    y_pred_scaled = model.predict(X)

    # 反归一化 ac_energy 到真实尺度
    ac_idx = list(feature_cols_all).index('ac_energy')
    ac_min = scaler.data_min_[ac_idx]
    ac_max = scaler.data_max_[ac_idx]
    y_pred = y_pred_scaled * (ac_max - ac_min) + ac_min

    return pd.Series(y_pred, index=d_lag.index)


if __name__ == '__main__':
    main()
