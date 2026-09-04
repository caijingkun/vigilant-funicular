# -*- coding: utf-8 -*-
"""
仿真对比实验模块
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统

对比两种运行模式：
  场景A（传统模式）：统一26°C设定、下课不关空调(空转)、不开窗
  场景B（AI优化模式）：AI预测 + 双约束决策算法动态调节

指标：
  - 总能耗 (kWh)
  - 理论节能率 (%)
  - CO₂超标时长占比 (%)
  - 理论碳减排量 (kg CO₂，按0.5 kg CO₂/kWh估算)

输出:
  results/experiment_report.txt       (实验报告)
  results/energy_comparison_bar.png  (能耗柱状图)
  results/energy_timeseries.png       (时间序列对比)
  results/co2_timeseries.png          (CO₂时间序列对比)
  results/metrics_summary.csv         (指标汇总)
=====================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                    'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_algorithm import EnergyDecisionMaker

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# 碳排放系数 kg CO₂ / kWh
CARBON_FACTOR = 0.5
# CO₂达标上限
CO2_LIMIT = 1000  # GB/T 18883-2022


# ============================================================
# 一、加载场景数据
# ============================================================
def load_data():
    """加载传统模式与优化模式数据集"""
    trad_path = os.path.join(DATA_DIR, 'classroom_energy_dataset.csv')
    opt_path  = os.path.join(DATA_DIR, 'classroom_energy_optimized.csv')

    df_trad = pd.read_csv(trad_path)
    df_trad['datetime'] = pd.to_datetime(df_trad['datetime'])
    df_opt = pd.read_csv(opt_path)
    df_opt['datetime'] = pd.to_datetime(df_opt['datetime'])

    return df_trad, df_opt


# ============================================================
# 二、AI优化模式仿真（用决策算法重新生成能耗）
# ============================================================
def simulate_ai_mode(df_opt):
    """
    使用data_generation.py生成的优化模式数据（基于物理仿真），
    叠加决策算法的推荐列用于展示。
    能耗/CO₂/温湿度均来自物理仿真，决策算法列用于展示AI建议。
    """
    dm = EnergyDecisionMaker()
    df = df_opt.copy()

    n = len(df)
    decision_setpoint = np.zeros(n)
    decision_window = np.zeros(n)
    decision_ac_on = np.zeros(n)
    decision_vent_min = np.zeros(n)
    decision_reason = [''] * n

    for i in range(n):
        row = df.iloc[i]
        decision = dm.make_decision(
            current_co2=row['co2'],
            outdoor_temp=row['outdoor_temp'],
            outdoor_humidity=row['outdoor_humidity'],
            indoor_temp=row['indoor_temp'],
            occupancy=row['occupancy'],
            is_class_time=bool(row['is_class_time']),
        )
        decision_setpoint[i] = decision.ac_setpoint
        decision_ac_on[i] = 1 if decision.ac_on else 0
        decision_window[i] = 1 if decision.open_window else 0
        decision_vent_min[i] = decision.ventilation_minutes

    # 叠加决策算法推荐列（展示用），能耗/CO₂/温湿度保持物理仿真值
    df['decision_setpoint'] = np.round(decision_setpoint, 1)
    df['decision_ac_on'] = decision_ac_on.astype(int)
    df['decision_window'] = decision_window.astype(int)
    df['decision_vent_min'] = decision_vent_min.astype(int)

    return df


# ============================================================
# 三、指标计算
# ============================================================
def calculate_metrics(df, label):
    """计算单场景指标"""
    total_energy = df['ac_energy'].sum()
    co2_exceed = (df['co2'] > CO2_LIMIT).sum()
    co2_exceed_pct = co2_exceed / len(df) * 100
    carbon = total_energy * CARBON_FACTOR
    avg_co2 = df['co2'].mean()
    max_co2 = df['co2'].max()
    comfort_pct = ((df['indoor_temp'] >= 24) & (df['indoor_temp'] <= 28)).sum() / len(df) * 100

    return {
        '模式': label,
        '总能耗(kWh)': round(total_energy, 1),
        'CO₂超标时长占比(%)': round(co2_exceed_pct, 2),
        '碳排放量(kg CO₂)': round(carbon, 1),
        'CO₂均值(ppm)': round(avg_co2, 0),
        'CO₂峰值(ppm)': round(max_co2, 0),
        '热舒适达标率(%)': round(comfort_pct, 2),
    }


# ============================================================
# 四、可视化
# ============================================================
def plot_bar_comparison(metrics_trad, metrics_ai):
    """能耗与碳排放柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['传统模式', 'AI优化模式']
    colors = ['#FF7043', '#4CAF50']

    # 总能耗
    vals = [metrics_trad['总能耗(kWh)'], metrics_ai['总能耗(kWh)']]
    axes[0].bar(labels, vals, color=colors, width=0.5)
    axes[0].set_title('总能耗对比', fontsize=13)
    axes[0].set_ylabel('能耗 (kWh)')
    for i, v in enumerate(vals):
        axes[0].text(i, v + max(vals)*0.01, f'{v:.0f}', ha='center', fontsize=12)

    # CO₂超标占比
    vals = [metrics_trad['CO₂超标时长占比(%)'], metrics_ai['CO₂超标时长占比(%)']]
    axes[1].bar(labels, vals, color=colors, width=0.5)
    axes[1].set_title('CO2超标时长占比', fontsize=13)
    axes[1].set_ylabel('占比 (%)')
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=12)

    # 碳排放
    vals = [metrics_trad['碳排放量(kg CO₂)'], metrics_ai['碳排放量(kg CO₂)']]
    axes[2].bar(labels, vals, color=colors, width=0.5)
    axes[2].set_title('碳排放量对比', fontsize=13)
    axes[2].set_ylabel('碳排放 (kg CO2)')
    for i, v in enumerate(vals):
        axes[2].text(i, v + max(vals)*0.01, f'{v:.0f}', ha='center', fontsize=12)

    plt.suptitle('传统模式 vs AI优化模式 对比', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'energy_comparison_bar.png'), dpi=150)
    plt.close()


def plot_timeseries_energy(df_trad, df_ai, hours=168):
    """能耗时间序列对比图（取前一周）"""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(hours)
    ax.plot(x, df_trad['ac_energy'].values[:hours],
            label='传统模式', color='#FF7043', linewidth=1.2)
    ax.plot(x, df_ai['ac_energy'].values[:hours],
            label='AI优化模式', color='#4CAF50', linewidth=1.2)
    ax.set_xlabel('时间（小时）')
    ax.set_ylabel('能耗 (kWh)')
    ax.set_title('一周能耗时间序列对比')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'energy_timeseries.png'), dpi=150)
    plt.close()


def plot_timeseries_co2(df_trad, df_ai, hours=168):
    """CO₂时间序列对比图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(hours)
    ax.plot(x, df_trad['co2'].values[:hours],
            label='传统模式', color='#FF7043', linewidth=1.2)
    ax.plot(x, df_ai['co2'].values[:hours],
            label='AI优化模式', color='#4CAF50', linewidth=1.2)
    ax.axhline(y=CO2_LIMIT, color='red', linestyle='--',
               label=f'CO2限值({CO2_LIMIT}ppm)', alpha=0.7)
    ax.set_xlabel('时间（小时）')
    ax.set_ylabel('CO2浓度 (ppm)')
    ax.set_title('一周室内CO2浓度对比')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'co2_timeseries.png'), dpi=150)
    plt.close()


# ============================================================
# 五、生成实验报告
# ============================================================
def generate_report(metrics_trad, metrics_ai):
    """生成文本实验报告"""
    energy_trad = metrics_trad['总能耗(kWh)']
    energy_ai   = metrics_ai['总能耗(kWh)']
    saving_rate = (1 - energy_ai / energy_trad) * 100
    energy_saved = energy_trad - energy_ai
    carbon_trad = metrics_trad['碳排放量(kg CO₂)']
    carbon_ai   = metrics_ai['碳排放量(kg CO₂)']
    carbon_reduced = carbon_trad - carbon_ai

    co2_trad_ok = 100 - metrics_trad['CO₂超标时长占比(%)']
    co2_ai_ok   = 100 - metrics_ai['CO₂超标时长占比(%)']

    report = []
    report.append("=" * 65)
    report.append("  仿真对比实验报告")
    report.append("  项目：基于AI的高校教室能耗预测与智能节能决策系统")
    report.append("  地区：桂林  |  数据：2024全年小时级仿真数据")
    report.append("=" * 65)
    report.append("")

    report.append("一、实验设计")
    report.append("-" * 40)
    report.append("场景A（传统模式）：统一26°C设定、下课不关空调(空转)、不开窗通风")
    report.append("场景B（AI优化模式）：AI预测 + 双约束决策算法动态调节")
    report.append("约束条件：①CO₂ ≤ 1000ppm(GB/T 18883-2022) ②室内温度24~28°C")
    report.append("碳排放系数：0.5 kg CO₂/kWh")
    report.append("")

    report.append("二、对比指标")
    report.append("-" * 40)
    report.append(f"{'指标':<25} {'传统模式':>12} {'AI优化模式':>12} {'改善':>12}")
    report.append("-" * 65)
    report.append(f"{'总能耗(kWh)':<25} {energy_trad:>12.1f} {energy_ai:>12.1f} "
                  f"{energy_saved:>12.1f}")
    report.append(f"{'节能率(%)':<25} {'—':>12} {'—':>12} {saving_rate:>12.1f}")
    report.append(f"{'碳排放量(kg CO₂)':<25} {carbon_trad:>12.1f} {carbon_ai:>12.1f} "
                  f"{carbon_reduced:>12.1f}")
    report.append(f"{'CO₂超标时长占比(%)':<25} "
                  f"{metrics_trad['CO₂超标时长占比(%)']:>12.2f} "
                  f"{metrics_ai['CO₂超标时长占比(%)']:>12.2f} "
                  f"{metrics_trad['CO₂超标时长占比(%)'] - metrics_ai['CO₂超标时长占比(%)']:>12.2f}")
    report.append(f"{'CO₂达标率(%)':<25} {co2_trad_ok:>12.2f} {co2_ai_ok:>12.2f} "
                  f"{co2_ai_ok - co2_trad_ok:>12.2f}")
    report.append(f"{'CO₂均值(ppm)':<25} "
                  f"{metrics_trad['CO₂均值(ppm)']:>12.0f} "
                  f"{metrics_ai['CO₂均值(ppm)']:>12.0f}")
    report.append(f"{'热舒适达标率(%)':<25} "
                  f"{metrics_trad['热舒适达标率(%)']:>12.2f} "
                  f"{metrics_ai['热舒适达标率(%)']:>12.2f}")
    report.append("")

    report.append("三、核心结论")
    report.append("-" * 40)
    report.append(f"1. 节能效果：AI优化模式较传统模式节能 {energy_saved:.1f} kWh，"
                  f"节能率 {saving_rate:.1f}%")
    report.append(f"2. 碳减排：理论碳减排量 {carbon_reduced:.1f} kg CO₂")
    report.append(f"3. 空气质量：CO₂达标率从 {co2_trad_ok:.1f}% 提升至 {co2_ai_ok:.1f}%")
    report.append(f"4. 热舒适：室内温度达标率从 {metrics_trad['热舒适达标率(%)']:.1f}% "
                  f"变化至 {metrics_ai['热舒适达标率(%)']:.1f}%")
    report.append(f"5. 节能策略核心：无课关空调 + 适宜天气自然通风 + 动态设定温度")
    report.append("")

    report.append("四、图表文件")
    report.append("-" * 40)
    report.append("results/energy_comparison_bar.png  — 能耗/CO₂/碳排放柱状图")
    report.append("results/energy_timeseries.png       — 一周能耗时间序列对比")
    report.append("results/co2_timeseries.png           — 一周CO₂时间序列对比")
    report.append("results/metrics_summary.csv          — 指标汇总数据")
    report.append("")
    report.append("=" * 65)

    report_text = '\n'.join(report)

    # 保存报告
    report_path = os.path.join(RESULT_DIR, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text, {
        'total_energy_trad': energy_trad,
        'total_energy_ai': energy_ai,
        'saving_rate': saving_rate,
        'carbon_reduced': carbon_reduced,
        'co2_compliance_trad': co2_trad_ok,
        'co2_compliance_ai': co2_ai_ok,
    }


# ============================================================
# 六、主函数
# ============================================================
def main():
    print("=" * 60)
    print("  仿真对比实验")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载场景数据...")
    df_trad, df_opt = load_data()
    print(f"  传统模式: {len(df_trad)} 行")
    print(f"  原始优化模式: {len(df_opt)} 行")

    # AI优化模式仿真（基于物理仿真数据 + 决策算法推荐列）
    print("[2/4] AI优化模式仿真（物理仿真 + 决策算法）...")
    df_ai = simulate_ai_mode(df_opt)
    print(f"  AI优化模式: {len(df_ai)} 行")

    # 指标计算
    print("[3/4] 计算对比指标...")
    metrics_trad = calculate_metrics(df_trad, '传统模式')
    metrics_ai   = calculate_metrics(df_ai, 'AI优化模式')

    print(f"\n  传统模式 —— 总能耗: {metrics_trad['总能耗(kWh)']:.1f} kWh, "
          f"CO₂超标: {metrics_trad['CO₂超标时长占比(%)']:.1f}%, "
          f"碳排放: {metrics_trad['碳排放量(kg CO₂)']:.1f} kg")
    print(f"  AI优化模式 —— 总能耗: {metrics_ai['总能耗(kWh)']:.1f} kWh, "
          f"CO₂超标: {metrics_ai['CO₂超标时长占比(%)']:.1f}%, "
          f"碳排放: {metrics_ai['碳排放量(kg CO₂)']:.1f} kg")

    # 生成图表
    print("[4/4] 生成图表与报告...")
    plot_bar_comparison(metrics_trad, metrics_ai)
    plot_timeseries_energy(df_trad, df_ai)
    plot_timeseries_co2(df_trad, df_ai)

    # 保存AI优化数据
    ai_path = os.path.join(DATA_DIR, 'classroom_energy_ai_optimized.csv')
    df_ai.to_csv(ai_path, index=False, encoding='utf-8-sig')

    # 保存指标汇总
    summary_df = pd.DataFrame([metrics_trad, metrics_ai])
    summary_path = os.path.join(RESULT_DIR, 'metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # 生成报告
    report_text, summary = generate_report(metrics_trad, metrics_ai)

    print("\n" + report_text)

    print(f"\n图表已保存至: {RESULT_DIR}/")
    print(f"AI优化数据已保存: {ai_path}")
    print("\n仿真对比实验完成！")


if __name__ == '__main__':
    main()
