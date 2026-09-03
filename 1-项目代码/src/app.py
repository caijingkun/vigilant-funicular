# -*- coding: utf-8 -*-
"""
Streamlit 可视化系统
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统
功能:
  - 总览仪表盘（核心指标 + 综合趋势）
  - 能耗预测（历史 + AI预测曲线）
  - 室内环境参数实时显示
  - AI智能调控建议（基于历史数据）
  - 实时调控建议（手动输入/上传CSV → 实时决策 + 批量导出）  ★新增
  - 传统模式 vs AI优化模式对比
配色: 蓝绿配色（贴合双碳主题）
启动: python run.py  或  streamlit run src/app.py
=====================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_algorithm import EnergyDecisionMaker

# 页面配置
st.set_page_config(
    page_title="教室能耗智能决策系统",
    page_icon="🌱",
    layout="wide"
)

# 配色
COLOR_PRIMARY = '#2196F3'   # 蓝色
COLOR_SECONDARY = '#4CAF50' # 绿色
COLOR_WARNING = '#FF9800'   # 橙色
COLOR_DANGER = '#F44336'    # 红色


# ============================================================
# 数据加载（缓存）
# ============================================================
@st.cache_data
def load_data():
    """加载所有数据集"""
    data = {}
    trad_path = os.path.join(DATA_DIR, 'classroom_energy_dataset.csv')
    opt_path = os.path.join(DATA_DIR, 'classroom_energy_optimized.csv')
    ai_path = os.path.join(DATA_DIR, 'classroom_energy_ai_optimized.csv')

    if os.path.exists(trad_path):
        data['traditional'] = pd.read_csv(trad_path)
        data['traditional']['datetime'] = pd.to_datetime(data['traditional']['datetime'])
    if os.path.exists(opt_path):
        data['optimized'] = pd.read_csv(opt_path)
        data['optimized']['datetime'] = pd.to_datetime(data['optimized']['datetime'])
    if os.path.exists(ai_path):
        data['ai'] = pd.read_csv(ai_path)
        data['ai']['datetime'] = pd.to_datetime(data['ai']['datetime'])

    summary_path = os.path.join(RESULT_DIR, 'metrics_summary.csv')
    if os.path.exists(summary_path):
        data['summary'] = pd.read_csv(summary_path)

    return data


# ============================================================
# 主界面
# ============================================================
def main():
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_SECONDARY});
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>🌱 基于AI的高校教室能耗预测与智能节能决策系统</h1>
            <p style='color: #E0F7FA; margin: 5px 0 0 0;'>桂林 · 多源数据融合AI预测 · 能耗与空气质量协同优化 · 轻量化无硬件改造</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("")

    data = load_data()

    # 侧边栏导航
    st.sidebar.markdown("### 📋 导航")
    page = st.sidebar.radio("选择页面", [
        "🎯 实时调控建议",
        "📊 总览仪表盘",
        "📈 能耗预测",
        "🌡️ 室内环境",
        "🤖 AI调控建议",
        "⚖️ 模式对比",
    ])

    if page == "🎯 实时调控建议":
        show_realtime_decision()
    elif page == "📊 总览仪表盘":
        show_dashboard(data)
    elif page == "📈 能耗预测":
        show_energy_prediction(data)
    elif page == "🌡️ 室内环境":
        show_indoor_environment(data)
    elif page == "🤖 AI调控建议":
        show_ai_decision(data)
    elif page == "⚖️ 模式对比":
        show_comparison(data)


# ============================================================
# 页面0: 实时调控建议（★新增核心功能）
# ============================================================
def show_realtime_decision():
    """实时调控建议页面：手动输入或上传CSV，生成决策建议"""
    st.header("🎯 实时调控建议")
    st.markdown("输入当前教室环境参数，AI决策算法将输出最优空调控制策略。")

    dm = EnergyDecisionMaker()

    # 选择输入方式
    input_mode = st.radio("选择输入方式", ["✍️ 手动输入", "📁 上传CSV批量处理"],
                         horizontal=True)

    if input_mode == "✍️ 手动输入":
        _show_manual_input(dm)
    else:
        _show_batch_input(dm)


def _show_manual_input(dm):
    """手动输入模式"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌡️ 温度参数")
        outdoor_temp = st.slider("室外温度 (°C)", -10.0, 45.0, 30.0, 0.5)
        indoor_temp = st.slider("室内温度 (°C)", 10.0, 40.0, 27.0, 0.5)
        outdoor_humidity = st.slider("室外湿度 (%)", 0, 100, 70)

    with col2:
        st.markdown("#### 💨 空气质量")
        co2 = st.slider("室内CO₂浓度 (ppm)", 300, 3000, 700, 10)
        st.caption(f"📊 标准限值: 1000ppm (GB/T 18883-2022)")
        co2_status = "✅ 达标" if co2 < 1000 else "⚠️ 超标"
        st.markdown(f"**当前状态**: {co2_status}")

    with col3:
        st.markdown("#### 👥 人员与课表")
        occupancy = st.slider("教室人数", 0, 80, 45)
        is_class_time = st.checkbox("当前为上课时段", value=True)
        st.caption("📊 上课时段: 8-12, 14-18（工作日）")

    # 生成建议按钮
    st.markdown("---")
    if st.button("🚀 生成调控建议", type="primary", use_container_width=True):
        decision = dm.make_decision(
            current_co2=co2,
            outdoor_temp=outdoor_temp,
            outdoor_humidity=outdoor_humidity,
            indoor_temp=indoor_temp,
            occupancy=occupancy,
            is_class_time=is_class_time,
        )

        _display_decision_card(decision, outdoor_temp, outdoor_humidity,
                               indoor_temp, co2, occupancy, is_class_time)


def _display_decision_card(decision, outdoor_temp, outdoor_humidity,
                           indoor_temp, co2, occupancy, is_class_time):
    """以卡片形式展示决策结果"""
    st.markdown("---")
    st.markdown("### 📋 AI调控建议结果")

    # 决策卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if decision.ac_on:
            st.metric("🌡️ 空调建议", f"设{decision.ac_setpoint:.0f}°C", delta="开启")
        else:
            st.metric("🌡️ 空调建议", "关闭", delta="节能模式")
        st.caption(f"传统模式: 26°C统一设定")

    with col2:
        if decision.open_window:
            st.metric("💨 开窗通风", "建议开窗", delta=f"{decision.ventilation_minutes}分钟")
        else:
            st.metric("💨 开窗通风", "不建议", delta=None)

    with col3:
        carbon_saved = decision.energy_saving * 0.5
        st.metric("⚡ 预计节能", f"{decision.energy_saving:.2f} kWh",
                  delta=f"{carbon_saved:.2f} kg CO₂减排")

    with col4:
        # CO₂状态
        if co2 < 600:
            co2_label = "✅ 良好"
        elif co2 < 1000:
            co2_label = "⚠️ 注意"
        else:
            co2_label = "🔴 超标"
        st.metric("🌬️ CO₂状态", f"{co2:.0f} ppm", delta=co2_label)

    # 能耗对比
    st.markdown("---")
    st.markdown("#### ⚡ 能耗对比分析")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("传统模式能耗", f"{decision.baseline_energy:.2f} kWh/h")
    with col_b:
        st.metric("AI优化能耗", f"{decision.optimized_energy:.2f} kWh/h")
    with col_c:
        saving_rate = (decision.energy_saving / max(decision.baseline_energy, 0.01)) * 100
        st.metric("节能率", f"{saving_rate:.1f}%")

    # 决策依据
    st.markdown("---")
    st.markdown("#### 📝 决策依据")
    st.info(f"**{decision.reason}**")

    if decision.detail_reasons:
        st.markdown("**详细分析：**")
        for detail in decision.detail_reasons:
            st.markdown(f"- {detail}")

    # 输入参数回显
    st.markdown("---")
    st.markdown("#### 📥 输入参数")
    input_df = pd.DataFrame({
        '参数': ['室外温度', '室外湿度', '室内温度', 'CO₂浓度', '教室人数', '是否上课'],
        '数值': [f"{outdoor_temp:.1f}°C", f"{outdoor_humidity:.0f}%",
                 f"{indoor_temp:.1f}°C", f"{co2:.0f}ppm",
                 f"{occupancy}人", "是" if is_class_time else "否"],
    })
    st.table(input_df)

    # 约束条件说明
    with st.expander("📋 约束条件说明"):
        st.markdown("""
        - ① **最小化空调能耗** — 在满足以下约束下选最低能耗方案
        - ② **CO₂ ≤ 1000 ppm** — GB/T 18883-2022 室内空气质量标准
        - ③ **室内温度 24~28°C** — 热舒适区间
        - ④ **碳排放系数** — 0.5 kg CO₂/kWh
        """)


def _show_batch_input(dm):
    """批量CSV上传模式"""
    st.markdown("#### 📁 上传教室环境数据CSV")

    # 模板下载提示
    template_path = os.path.join(DATA_DIR, 'input_template.csv')
    if os.path.exists(template_path):
        st.success("✅ 数据模板已就绪，请按模板格式填写数据后上传。")
        with open(template_path, 'r', encoding='utf-8-sig') as f:
            st.download_button(
                label="📥 下载模板文件",
                data=f.read(),
                file_name="input_template.csv",
                mime="text/csv"
            )

    st.markdown("**CSV格式要求：**")
    st.markdown("""
    列名: `datetime, outdoor_temp, outdoor_humidity, indoor_temp, indoor_humidity, co2, occupancy, is_class_time`
    
    - datetime: 日期时间 (如 2024-07-15 08:00:00)
    - outdoor_temp: 室外温度 (°C)
    - outdoor_humidity: 室外湿度 (%)
    - indoor_temp: 室内温度 (°C)
    - indoor_humidity: 室内湿度 (%)
    - co2: CO₂浓度 (ppm)
    - occupancy: 教室人数
    - is_class_time: 是否上课 (1=是, 0=否)
    """)

    uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 文件加载成功，共 {len(df)} 行数据")

            # 验证列名
            required_cols = ['outdoor_temp', 'outdoor_humidity', 'indoor_temp',
                           'co2', 'occupancy', 'is_class_time']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ 缺少必要列: {missing}")
                return

            # 预览数据
            st.markdown("#### 📊 数据预览")
            st.dataframe(df.head(10), use_container_width=True)

            # 生成批量决策
            if st.button("🚀 生成全天调控建议", type="primary"):
                with st.spinner("正在生成AI决策建议..."):
                    # 补充 ac_energy 列（如果缺少）
                    if 'ac_energy' not in df.columns:
                        df['ac_energy'] = 0.0

                    result_df = dm.batch_decide(df)

                st.success(f"✅ 决策生成完成！共 {len(result_df)} 条建议")

                # 显示结果
                st.markdown("#### 📋 调控建议表")
                display_cols = ['datetime', 'outdoor_temp', 'indoor_temp', 'co2',
                              'occupancy', '建议设定温度', '是否开空调',
                              '是否开窗通风', '通风时长(分钟)', '预计节能(kWh)']
                available_cols = [c for c in display_cols if c in result_df.columns]
                st.dataframe(result_df[available_cols], use_container_width=True)

                # 统计摘要
                st.markdown("#### 📊 统计摘要")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总节能", f"{result_df['预计节能(kWh)'].sum():.1f} kWh")
                with col2:
                    st.metric("碳减排", f"{result_df['预计节能(kWh)'].sum() * 0.5:.1f} kg CO₂")
                with col3:
                    ac_on_pct = (result_df['是否开空调'] == '是').sum() / len(result_df) * 100
                    st.metric("空调开启占比", f"{100-ac_on_pct:.0f}% (关闭)")
                with col4:
                    vent_pct = (result_df['是否开窗通风'] == '是').sum() / len(result_df) * 100
                    st.metric("建议通风占比", f"{vent_pct:.0f}%")

                # 导出CSV
                st.markdown("---")
                csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 导出调控建议表 (CSV)",
                    data=csv,
                    file_name="ai_decision_suggestions.csv",
                    mime="text/csv",
                    type="primary"
                )

                # 可视化
                st.markdown("#### 📈 全天决策可视化")
                _plot_batch_decisions(result_df)

        except Exception as e:
            st.error(f"❌ 文件解析失败: {e}")
            st.info("请确保CSV格式正确，可下载模板文件参考。")


def _plot_batch_decisions(df):
    """绘制批量决策可视化图"""
    n = len(df)
    x = range(n)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 温度与建议
    if 'datetime' in df.columns:
        x_labels = df['datetime'].astype(str).str[:16]  # 截断到分钟
    else:
        x_labels = list(x)

    axes[0].bar(x, df['建议设定温度'], color=COLOR_SECONDARY, alpha=0.7, label='建议设定温度')
    if 'indoor_temp' in df.columns:
        axes[0].plot(x, df['indoor_temp'], color=COLOR_PRIMARY, linewidth=1, label='室内温度')
        axes[0].plot(x, df.get('outdoor_temp', pd.Series(dtype=float)), color=COLOR_WARNING,
                     linewidth=1, alpha=0.6, label='室外温度')
    axes[0].axhspan(24, 28, alpha=0.1, color=COLOR_SECONDARY)
    axes[0].set_ylabel('温度 (°C)')
    axes[0].set_title('温度与空调建议')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_ylim(0, 40)

    # CO₂
    axes[1].plot(x, df['co2'], color=COLOR_DANGER, linewidth=1.5)
    axes[1].axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='CO₂限值(1000ppm)')
    axes[1].axhline(y=800, color=COLOR_WARNING, linestyle=':', alpha=0.5, label='预警值(800ppm)')
    axes[1].set_ylabel('CO₂ (ppm)')
    axes[1].set_title('室内CO₂浓度')
    axes[1].legend(loc='upper right', fontsize=8)

    # 节能
    axes[2].bar(x, df['预计节能(kWh)'], color=COLOR_SECONDARY, alpha=0.7, label='预计节能')
    axes[2].set_ylabel('节能 (kWh)')
    axes[2].set_title('预计节能效果')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ============================================================
# 页面1: 总览仪表盘
# ============================================================
def show_dashboard(data):
    st.header("📊 总览仪表盘")

    df = data.get('traditional')
    if df is None:
        st.warning("数据未加载，请先运行数据生成模块")
        st.code("python run.py", language='bash')
        return

    summary = data.get('summary')

    # 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)

    total_energy = df['ac_energy'].sum()
    with col1:
        st.metric("传统模式总能耗", f"{total_energy:.0f} kWh",
                  delta=f"{total_energy*0.5:.0f} kg CO₂")

    if 'ai' in data:
        ai_energy = data['ai']['ac_energy'].sum()
        saving_rate = (1 - ai_energy / total_energy) * 100 if total_energy > 0 else 0
        with col2:
            st.metric("AI优化模式能耗", f"{ai_energy:.0f} kWh",
                      delta=f"↓{saving_rate:.1f}%", delta_color="inverse")
        with col3:
            st.metric("理论节能量", f"{total_energy - ai_energy:.0f} kWh",
                      delta=f"{(total_energy-ai_energy)*0.5:.0f} kg CO₂减排")
        with col4:
            co2_avg = data['ai']['co2'].mean()
            st.metric("AI模式平均CO₂", f"{co2_avg:.0f} ppm",
                      delta="达标" if co2_avg < 1000 else "超标")
    else:
        with col2:
            st.metric("室外温度范围",
                      f"{df['outdoor_temp'].min():.0f}~{df['outdoor_temp'].max():.0f}°C")
        with col3:
            st.metric("CO₂范围",
                      f"{df['co2'].min():.0f}~{df['co2'].max():.0f} ppm")
        with col4:
            st.metric("数据行数", f"{len(df)} 行")

    st.markdown("---")

    # 数据范围选择
    col_a, col_b = st.columns(2)
    with col_a:
        start_idx = st.slider("选择查看起始位置（小时）", 0, len(df)-168, 0)
    with col_b:
        window = st.selectbox("查看窗口", [24, 48, 72, 168],
                              format_func=lambda x: f"{x}小时({x//24}天)")

    end_idx = min(start_idx + window, len(df))
    df_view = df.iloc[start_idx:end_idx]

    # 综合趋势图
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df_view['datetime'], df_view['ac_energy'],
                 color=COLOR_PRIMARY, linewidth=1)
    axes[0].set_ylabel('能耗 (kWh)')
    axes[0].set_title('空调能耗')
    axes[0].fill_between(df_view['datetime'], 0, df_view['ac_energy'], alpha=0.2, color=COLOR_PRIMARY)

    axes[1].plot(df_view['datetime'], df_view['indoor_temp'],
                 color=COLOR_SECONDARY, linewidth=1, label='室内温度')
    axes[1].plot(df_view['datetime'], df_view['outdoor_temp'],
                 color=COLOR_WARNING, linewidth=1, alpha=0.6, label='室外温度')
    axes[1].axhspan(24, 28, alpha=0.15, color=COLOR_SECONDARY, label='舒适区间')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].set_title('室内外温度')
    axes[1].legend(loc='upper right', fontsize=8)

    axes[2].plot(df_view['datetime'], df_view['co2'],
                 color=COLOR_DANGER, linewidth=1)
    axes[2].axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='CO2限值(1000ppm)')
    axes[2].set_ylabel('CO2 (ppm)')
    axes[2].set_title('室内CO2浓度')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ============================================================
# 页面2: 能耗预测
# ============================================================
def show_energy_prediction(data):
    st.header("📈 能耗预测")

    df = data.get('traditional')
    if df is None:
        st.warning("数据未加载")
        return

    st.markdown("**RandomForest 模型预测空调能耗**")
    st.markdown("输入特征：课表强度指数、室外温湿度、室内温湿度、CO₂、过去24小时能耗")
    st.info("本模型为单步预测（用当前已知特征 + 过去24h能耗预测当前能耗）。"
            "下方在历史数据上做预测并与真实值对比，直观展示模型精度。")

    # 加载模型并对全量数据预测（复用训练时的预处理，保证特征对齐）
    from model_training import predict_ac_energy
    pred_all = predict_ac_energy(df)
    if pred_all is None:
        st.warning("未找到训练好的模型，请先运行 `python run.py` 完成训练")
        return

    # 选择查看日期（从第2天起，确保有前24小时历史用于滞后特征）
    day_idx = st.slider("选择日期 (第几天)", 2, len(df)//24, 2)
    day_start = (day_idx - 1) * 24
    day_end = day_start + 24

    # 对齐：y_true 取原始行 [day_start, day_end)，y_pred 用标签索引取同范围
    y_true = df['ac_energy'].iloc[day_start:day_end]
    y_pred = pred_all.loc[day_start:day_end - 1]
    x = df['datetime'].iloc[day_start:day_end]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y_true, color=COLOR_PRIMARY, linewidth=1.5, label='真实能耗')
    ax.plot(x, y_pred, color=COLOR_SECONDARY, linewidth=1.5, linestyle='--',
            label='模型预测')
    ax.set_xlabel('时间')
    ax.set_ylabel('能耗 (kWh)')
    ax.set_title(f'能耗预测 vs 真实值 — 第{day_idx}天')
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 当日预测精度
    from sklearn.metrics import r2_score, mean_absolute_error
    day_r2 = r2_score(y_true, y_pred)
    day_mae = mean_absolute_error(y_true, y_pred)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("当日预测 R²", f"{day_r2:.3f}")
    with col2:
        st.metric("当日平均绝对误差 (MAE)", f"{day_mae:.3f} kWh")

    # 预测精度指标
    metrics_path = os.path.join(RESULT_DIR, 'model_metrics.txt')
    if os.path.exists(metrics_path):
        st.markdown("#### 模型精度指标")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            st.text(f.read())

    # 预测对比图
    pred_path = os.path.join(RESULT_DIR, 'prediction_vs_actual.png')
    if os.path.exists(pred_path):
        st.image(pred_path, caption="测试集模型预测 vs 真实值")


# ============================================================
# 页面3: 室内环境
# ============================================================
def show_indoor_environment(data):
    st.header("🌡️ 室内环境参数")

    df = data.get('traditional')
    if df is None:
        st.warning("数据未加载")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        hour_idx = st.slider("选择小时（从第1天起）", 0, len(df)-1, 0)
    with col2:
        st.markdown(f"**当前时刻**: {df.iloc[hour_idx]['datetime']}")

    row = df.iloc[hour_idx]

    # 实时参数卡片
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        temp_color = COLOR_SECONDARY if 24 <= row['indoor_temp'] <= 28 else COLOR_WARNING
        st.metric("室内温度", f"{row['indoor_temp']:.1f} °C")
        st.markdown(f"舒适区间: 24-28°C | 室外: {row['outdoor_temp']:.1f}°C")
    with col_b:
        st.metric("室内湿度", f"{row['indoor_humidity']:.0f} %")
        st.markdown(f"室外湿度: {row['outdoor_humidity']:.0f}%")
    with col_c:
        co2 = row['co2']
        co2_status = "✅ 达标" if co2 < 1000 else "⚠️ 超标"
        st.metric("室内CO₂", f"{co2:.0f} ppm", delta=co2_status)
        st.markdown(f"限值: 1000ppm (GB/T 18883-2022) | 人数: {row['occupancy']}")

    # 24小时趋势
    start = max(0, hour_idx - 12)
    end = min(len(df), hour_idx + 12)
    df_view = df.iloc[start:end]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df_view['datetime'], df_view['indoor_temp'], color=COLOR_PRIMARY)
    axes[0, 0].axhspan(24, 28, alpha=0.15, color=COLOR_SECONDARY)
    axes[0, 0].set_title('室内温度')
    axes[0, 0].set_ylabel('°C')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=30)

    axes[0, 1].plot(df_view['datetime'], df_view['indoor_humidity'], color=COLOR_SECONDARY)
    axes[0, 1].set_title('室内湿度')
    axes[0, 1].set_ylabel('%')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=30)

    axes[1, 0].plot(df_view['datetime'], df_view['co2'], color=COLOR_DANGER)
    axes[1, 0].axhline(y=1000, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('CO2浓度')
    axes[1, 0].set_ylabel('ppm')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=30)

    axes[1, 1].bar(range(len(df_view)), df_view['occupancy'], color=COLOR_WARNING)
    axes[1, 1].set_title('教室人数')
    axes[1, 1].set_ylabel('人')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ============================================================
# 页面4: AI调控建议（基于历史数据）
# ============================================================
def show_ai_decision(data):
    st.header("🤖 AI智能调控建议")

    df = data.get('traditional')
    if df is None:
        st.warning("数据未加载")
        return

    dm = EnergyDecisionMaker()

    col1, col2 = st.columns([1, 3])
    with col1:
        hour_idx = st.slider("选择小时", 0, len(df)-1, 0)
    with col2:
        st.markdown(f"**时刻**: {df.iloc[hour_idx]['datetime']}")

    row = df.iloc[hour_idx]

    # 获取决策
    decision = dm.make_decision(
        current_co2=row['co2'],
        outdoor_temp=row['outdoor_temp'],
        outdoor_humidity=row['outdoor_humidity'],
        indoor_temp=row['indoor_temp'],
        occupancy=row['occupancy'],
        is_class_time=bool(row['is_class_time']),
    )

    # 决策卡片
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if decision.ac_on:
            st.metric("空调建议", f"设{decision.ac_setpoint}°C", delta="开启")
        else:
            st.metric("空调建议", "关闭", delta="节能")
    with col_b:
        vent_text = f"开{decision.ventilation_minutes}分钟" if decision.open_window else "否"
        st.metric("开窗通风", "是" if decision.open_window else "否",
                  delta=vent_text if decision.open_window else None)
    with col_c:
        st.metric("预计节能", f"{decision.energy_saving:.2f} kWh",
                  delta=f"{decision.energy_saving*0.5:.2f} kg CO₂")
    with col_d:
        co2_status = "✅达标" if row['co2'] < 1000 else "⚠️超标"
        st.metric("CO₂状态", f"{row['co2']:.0f} ppm", delta=co2_status)

    # 决策依据
    st.info(f"📋 **决策依据**: {decision.reason}")

    # 当前环境状态
    st.markdown("#### 当前环境输入")
    env_df = pd.DataFrame({
        '参数': ['室外温度', '室外湿度', '室内温度', '室内湿度', 'CO₂浓度',
                '教室人数', '是否上课', 'AI预测能耗'],
        '数值': [f"{row['outdoor_temp']:.1f}°C", f"{row['outdoor_humidity']:.0f}%",
                 f"{row['indoor_temp']:.1f}°C", f"{row['indoor_humidity']:.0f}%",
                 f"{row['co2']:.0f}ppm", f"{row['occupancy']}人",
                 "是" if row['is_class_time'] else "否",
                 f"{row['ac_energy']:.2f}kWh"],
    })
    st.table(env_df)

    # 约束条件说明
    st.markdown("#### 约束条件")
    st.markdown("""
    - ① **最小化空调能耗** — 在满足以下约束下选最低能耗方案
    - ② **CO₂ ≤ 1000 ppm** — GB/T 18883-2022 室内空气质量标准
    - ③ **室内温度 24~28°C** — 热舒适区间
    """)

    # 批量决策时间序列
    st.markdown("---")
    st.markdown("#### 24小时AI决策序列")
    start = max(0, hour_idx - 12)
    end = min(len(df), hour_idx + 12)
    df_view = df.iloc[start:end].copy()

    decisions_list = []
    for _, r in df_view.iterrows():
        d = dm.make_decision(
            current_co2=r['co2'],
            outdoor_temp=r['outdoor_temp'],
            outdoor_humidity=r['outdoor_humidity'],
            indoor_temp=r['indoor_temp'],
            occupancy=r['occupancy'],
            is_class_time=bool(r['is_class_time']),
        )
        decisions_list.append({
            '建议设定温度': d.ac_setpoint if d.ac_on else 0,
            '是否开窗': 1 if d.open_window else 0,
            '通风时长(分)': d.ventilation_minutes,
        })
    dec_df = pd.DataFrame(decisions_list)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(range(len(dec_df)), dec_df['建议设定温度'], color=COLOR_SECONDARY, alpha=0.7, label='建议温度')
    ax1.set_ylabel('建议温度 (°C)', color=COLOR_SECONDARY)
    ax1.set_ylim(0, 30)
    ax2 = ax1.twinx()
    ax2.plot(range(len(dec_df)), dec_df['是否开窗']*10, 'o-', color=COLOR_PRIMARY, label='开窗(0/10)', markersize=3)
    ax2.set_ylabel('开窗通风', color=COLOR_PRIMARY)
    plt.title('AI调控建议序列')
    fig.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ============================================================
# 页面5: 模式对比
# ============================================================
def show_comparison(data):
    st.header("⚖️ 传统模式 vs AI优化模式")

    df_trad = data.get('traditional')
    df_ai = data.get('ai')

    if df_trad is None:
        st.warning("数据未加载")
        return

    # 核心对比指标
    summary = data.get('summary')
    if summary is not None:
        st.markdown("#### 核心指标对比")
        st.dataframe(summary, use_container_width=True)

    energy_trad = df_trad['ac_energy'].sum()
    if df_ai is not None:
        energy_ai = df_ai['ac_energy'].sum()
        saving_rate = (1 - energy_ai / energy_trad) * 100 if energy_trad > 0 else 0
        carbon_saved = (energy_trad - energy_ai) * 0.5

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("节能率", f"{saving_rate:.1f}%")
        with col2:
            st.metric("节能量", f"{energy_trad - energy_ai:.0f} kWh")
        with col3:
            st.metric("碳减排", f"{carbon_saved:.0f} kg CO₂")

    # 时间范围选择
    days = st.slider("选择对比天数（从第1天起）", 1, min(30, len(df_trad)//24), 7)
    hours = days * 24

    df_t = df_trad.iloc[:hours]
    if df_ai is not None:
        df_a = df_ai.iloc[:hours]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 能耗
    axes[0].plot(df_t['datetime'], df_t['ac_energy'],
                 color='#FF7043', linewidth=1, label='传统模式')
    if df_ai is not None:
        axes[0].plot(df_a['datetime'], df_a['ac_energy'],
                     color=COLOR_SECONDARY, linewidth=1, label='AI优化模式')
    axes[0].set_ylabel('能耗 (kWh)')
    axes[0].set_title('空调能耗对比')
    axes[0].legend(fontsize=9)

    # CO₂
    axes[1].plot(df_t['datetime'], df_t['co2'],
                 color='#FF7043', linewidth=1, label='传统模式')
    if df_ai is not None:
        axes[1].plot(df_a['datetime'], df_a['co2'],
                     color=COLOR_SECONDARY, linewidth=1, label='AI优化模式')
    axes[1].axhline(y=1000, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('CO2 (ppm)')
    axes[1].set_title('室内CO2浓度对比')
    axes[1].legend(fontsize=9)

    # 室内温度
    axes[2].plot(df_t['datetime'], df_t['indoor_temp'],
                 color='#FF7043', linewidth=1, label='传统模式')
    if df_ai is not None:
        axes[2].plot(df_a['datetime'], df_a['indoor_temp'],
                     color=COLOR_SECONDARY, linewidth=1, label='AI优化模式')
    axes[2].axhspan(24, 28, alpha=0.1, color=COLOR_SECONDARY)
    axes[2].set_ylabel('温度 (°C)')
    axes[2].set_title('室内温度对比')
    axes[2].legend(fontsize=9)

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 柱状对比图
    bar_path = os.path.join(RESULT_DIR, 'energy_comparison_bar.png')
    if os.path.exists(bar_path):
        st.image(bar_path, caption="能耗/CO₂/碳排放综合对比")

    # 实验报告
    report_path = os.path.join(RESULT_DIR, 'experiment_report.txt')
    if os.path.exists(report_path):
        st.markdown("#### 实验报告")
        with open(report_path, 'r', encoding='utf-8') as f:
            st.text(f.read())


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    main()
