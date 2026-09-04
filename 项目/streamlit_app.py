# -*- coding: utf-8 -*-
"""
Streamlit Cloud 部署入口文件
=====================================================
作用：直接放在项目根目录，供 https://share.streamlit.io/ 作为启动入口

Streamlit Cloud 部署要求：
  1. 入口必须是一个可直接被 `streamlit run xxx.py` 调用的 Python 脚本
     （不能是用 subprocess 再去启动另一个 streamlit 进程的 run.py）
  2. 本文件首次启动时会：自动生成数据 → 自动训练模型 → 再渲染可视化界面
  3. 所有依赖都通过根目录 requirements.txt 安装（Streamlit Cloud 自动处理）
  4. 适用于 Linux 容器环境（Streamlit Cloud 运行环境）

使用方式（Streamlit Cloud）：
  - App file path:  streamlit_app.py
=====================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 路径配置（必须最先处理，保证后续 import 与数据读写不报错）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

for _dir in [SRC_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# 把 src 加到 sys.path 以便 import 子模块
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ============================================================
# 2. 适配 Linux 环境下 matplotlib 中文字体（Streamlit Cloud 是 Ubuntu）
#    Windows 上常见的 SimHei / Microsoft YaHei 在 Linux 不存在
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def _configure_chinese_font():
    """尝试找到可用中文字体；找不到则退回英文并保证不报错。"""
    # 常见 Linux/Streamlit Cloud 中可能存在的中文字体候选
    font_candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Source Han Sans CN',
        'AR PL UMing CN',
        'Droid Sans Fallback',
        'DejaVu Sans',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        # 真的没有中文字体就用 DejaVu Sans（至少英文不乱码，中文用方框也比崩好）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

_configure_chinese_font()

# ============================================================
# 3. 数据自动生成 / 模型自动训练
#    Streamlit Cloud 冷启动时 data/ 与 models/ 目录可能为空
#    这里使用 @st.cache_resource 保证只在首次访问执行一次
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np


def _ensure_data_files():
    """确保三个核心 CSV 文件存在；缺哪个就自动调用生成模块生成。"""
    trad_csv = os.path.join(DATA_DIR, 'classroom_energy_dataset.csv')
    opt_csv = os.path.join(DATA_DIR, 'classroom_energy_optimized.csv')
    ai_csv = os.path.join(DATA_DIR, 'classroom_energy_ai_optimized.csv')

    all_exist = (
        os.path.exists(trad_csv) and os.path.getsize(trad_csv) > 1000
        and os.path.exists(opt_csv) and os.path.getsize(opt_csv) > 1000
        and os.path.exists(ai_csv) and os.path.getsize(ai_csv) > 1000
    )
    if all_exist:
        return True

    try:
        from data_generation import main as gen_main
        gen_main()
        return True
    except Exception as e:
        st.error(f"❌ 数据生成失败: {e}")
        return False


def _ensure_model_files():
    """确保训练好的模型存在；不存在则自动训练。"""
    rf_pkl = os.path.join(MODEL_DIR, 'rf_model.pkl')
    energy_pkl = os.path.join(MODEL_DIR, 'energy_model.pkl')
    info_pkl = os.path.join(MODEL_DIR, 'model_info.pkl')

    ok_rf = os.path.exists(rf_pkl) and os.path.getsize(rf_pkl) > 1000
    ok_en = os.path.exists(energy_pkl) and os.path.getsize(energy_pkl) > 1000
    ok_info = os.path.exists(info_pkl)

    if ok_rf and ok_en and ok_info:
        return True

    try:
        from model_training import main as train_main
        train_main()
        return True
    except Exception as e:
        st.warning(f"⚠️ 模型训练失败: {e}\n（能耗预测页将不可用，但其他功能不受影响）")
        return False


def _ensure_sim_results():
    """确保仿真实验结果（metrics_summary.csv 等）存在。"""
    summary_csv = os.path.join(RESULT_DIR, 'metrics_summary.csv')
    if os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 100:
        return True
    try:
        from simulation_experiment import main as sim_main
        sim_main()
        return True
    except Exception as e:
        return False


def _ensure_input_template():
    """生成数据录入模板 CSV（首次运行用）。"""
    tmpl = os.path.join(DATA_DIR, 'input_template.csv')
    if os.path.exists(tmpl):
        return
    dates = pd.date_range('2024-07-15 00:00:00', periods=24, freq='h')
    df = pd.DataFrame({
        'datetime': dates.strftime('%Y-%m-%d %H:%M:%S'),
        'outdoor_temp': [24,23,22,22,22,23,25,27,29,31,33,34,35,35,34,33,31,29,28,27,26,25,25,24],
        'outdoor_humidity': [75,76,77,78,78,77,74,70,66,62,58,55,53,52,54,57,62,66,69,71,72,73,74,75],
        'indoor_temp': [26,26,25,25,25,26,27,28,28,29,30,30,30,30,29,28,28,27,27,27,26,26,26,26],
        'indoor_humidity': [65,65,66,66,66,65,64,62,60,58,56,55,54,54,55,57,59,61,62,63,64,64,65,65],
        'co2': [450,440,430,425,420,415,500,800,950,1100,1200,1250,1300,1280,1150,900,700,650,550,500,480,470,460,455],
        'occupancy': [0,0,0,0,0,0,10,45,50,50,50,50,50,45,0,0,40,45,50,50,30,10,0,0],
        'is_class_time': [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0],
    })
    df.to_csv(tmpl, index=False, encoding='utf-8-sig')


@st.cache_resource(show_spinner=False)
def bootstrap_environment():
    """
    冷启动引导：只在第一次被访问时执行一次，后续命中缓存直接返回。
    返回 True 表示基础环境 OK（即使模型训练失败，只要算法决策模块 OK 就算 OK）。
    """
    steps = [
        ("📊 生成仿真数据集", _ensure_data_files),
        ("🤖 训练 AI 预测模型", _ensure_model_files),
        ("🧪 生成仿真对比结果", _ensure_sim_results),
        ("📋 生成数据录入模板", lambda: (_ensure_input_template(), True)[1]),
    ]
    all_ok = True
    for label, fn in steps:
        try:
            ok = fn()
            if not ok:
                all_ok = False
        except Exception as e:
            all_ok = False
    return all_ok


# ============================================================
# 4. 页面配置（必须在任何 st.xxx 输出之前）
# ============================================================
st.set_page_config(
    page_title="教室能耗智能决策系统",
    page_icon="🌱",
    layout="wide",
)

# 引导过程放在侧边栏顶部显示，不干扰主界面
with st.sidebar:
    with st.spinner("🔧 正在初始化系统（首次访问约需 1-2 分钟）..."):
        bootstrap_environment()
    st.success("✅ 系统初始化完成")

# ============================================================
# 5. 加载并运行 src/app.py 中的主界面
#    这里直接 import app.main 然后调用即可，避免再启动子进程
# ============================================================
import app as _app

if __name__ == '__main__':
    _app.main()
