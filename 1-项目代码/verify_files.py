# -*- coding: utf-8 -*-
"""验证脚本：检查所有输出文件"""
import os, joblib

BASE = os.path.dirname(os.path.abspath(__file__))

print("=== models/ 目录文件 ===")
model_dir = os.path.join(BASE, "models")
for f in sorted(os.listdir(model_dir)):
    size = os.path.getsize(os.path.join(model_dir, f))
    print(f"  {f}: {size:,} bytes")

print("\n=== results/ 目录文件 ===")
result_dir = os.path.join(BASE, "results")
for f in sorted(os.listdir(result_dir)):
    size = os.path.getsize(os.path.join(result_dir, f))
    print(f"  {f}: {size:,} bytes")

print("\n=== data/ 目录文件 ===")
data_dir = os.path.join(BASE, "data")
for f in sorted(os.listdir(data_dir)):
    size = os.path.getsize(os.path.join(data_dir, f))
    print(f"  {f}: {size:,} bytes")

print("\n=== 验证模型文件可加载 ===")
info = joblib.load(os.path.join(model_dir, "model_info.pkl"))
print(f"  model_type: {info['model_type']}")
print(f"  lookback: {info['lookback']}")
print(f"  feature数量: {len(info['feature_cols'])}")

rf_data = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
model = rf_data["model"]
print(f"  RandomForest: n_estimators={model.n_estimators}, max_depth={model.max_depth}")

print("\n=== 验证预测图表非空 ===")
from PIL import Image
import io
chart_path = os.path.join(result_dir, "prediction_vs_actual.png")
img = Image.open(chart_path)
print(f"  prediction_vs_actual.png: {img.size[0]}x{img.size[1]} pixels")
print("\n所有文件验证完成！")
