# -*- coding: utf-8 -*-
"""
数据生成模块 —— 仿真多源数据集生成
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统
功能: 生成桂林地区高校教室的小时级仿真数据集，包含：
  - 桂林气象数据（温度、湿度）—— 一整年小时级
  - 课表特征模拟（上课/下课/周末）
  - 室内CO₂浓度模拟（基于人员数和通风量）
  - 空调能耗模拟（传统模式 vs 优化模式）
  - 室内温湿度模拟
输出: data/classroom_energy_dataset.csv
=====================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

# ============================================================
# 一、全局参数配置
# ============================================================

# --- 教室物理参数 ---
ROOM_L, ROOM_W, ROOM_H = 9.0, 6.0, 3.0          # 长/宽/高 (m)
ROOM_VOLUME = ROOM_L * ROOM_W * ROOM_H          # 体积 162 m³
ROOM_AREA   = ROOM_L * ROOM_W                   # 面积 54 m²
WALL_AREA   = 2 * (ROOM_L + ROOM_W) * ROOM_H    # 外墙面积 90 m²
WINDOW_AREA = WALL_AREA * 0.20                   # 窗户面积 18 m²
ROOF_AREA   = ROOM_AREA                          # 屋顶面积 54 m²
MAX_CAPACITY = 60                                # 最大容纳人数

# --- 空调参数 ---
AC_NUM = 2                         # 空调台数
AC_POWER_PER = 2.5                 # 单台额定功率 kW
AC_COP = 3.2                       # 能效比
AC_MAX_COOLING = AC_POWER_PER * AC_COP * AC_NUM  # 最大制冷量 16 kW
AC_STANDBY_POWER = 0.3             # 待机功率 kW

# --- CO₂与通风参数 ---
CO2_OUTDOOR = 420                  # 室外CO₂浓度 ppm
CO2_PER_PERSON_LPS = 0.005        # 每人CO₂生成率 L/s（坐姿轻活动）
ACH_CLOSED = 0.5                  # 关门窗换气次数 次/h
CH_WINDOW  = 6.0                  # 开窗换气次数 次/h
ACH_AC     = 1.2                  # 空调新风换气次数 次/h

# --- 热学参数 ---
U_WALL   = 1.5                     # 外墙传热系数 W/(m²·K)
U_ROOF   = 1.0                     # 屋顶传热系数
U_WINDOW = 5.8                     # 窗户传热系数
AIR_DENSITY = 1.2                  # 空气密度 kg/m³
AIR_CP = 1.005                     # 空气定压比热 kJ/(kg·K)
PEOPLE_HEAT = 0.10                 # 每人显热 kW
EQUIP_HEAT_CLASS = 1.2            # 有课时设备/照明热负荷 kW
EQUIP_HEAT_IDLE  = 0.2            # 无课时设备热负荷 kW

# --- 时间范围 ---
START_DATE = datetime(2024, 1, 1, 0)
END_DATE   = datetime(2024, 12, 31, 23)  # 全年8760小时

# --- 课表时段（小时） ---
CLASS_PERIODS = [
    (8, 10),   # 第一节 08:00-10:00
    (10, 12),  # 第二节 10:00-12:00
    (14, 16),  # 第三节 14:00-16:00
    (16, 18),  # 第四节 16:00-18:00
]


# ============================================================
# 二、桂林气象数据生成
# ============================================================
def generate_weather(start, end):
    """
    生成桂林地区小时级气象数据（温度、湿度）。
    桂林属亚热带季风气候：夏季28-35°C，冬季8-15°C，春秋过渡。
    """
    dates = pd.date_range(start, end, freq='h')
    n = len(dates)
    months = dates.month
    hours  = dates.hour

    # 各月平均温度基准
    month_temp = {1:9, 2:11, 3:15, 4:20, 5:24, 6:27,
                  7:30, 8:30, 9:26, 10:21, 11:15, 12:10}
    # 各月平均湿度基准
    month_hum  = {1:70, 2:72, 3:75, 4:78, 5:80, 6:82,
                  7:80, 8:78, 9:72, 10:68, 11:68, 12:70}

    temp = np.zeros(n)
    hum  = np.zeros(n)
    for i in range(n):
        m, h = months[i], hours[i]
        # 日变化：最低温6:00，最高温14:00，振幅5°C
        daily_t = 5.0 * np.sin(2 * np.pi * (h - 9) / 24)
        temp[i] = month_temp[m] + daily_t + np.random.normal(0, 1.5)
        # 湿度与温度反相关
        daily_h = -8.0 * np.sin(2 * np.pi * (h - 9) / 24)
        hum[i]  = month_hum[m] + daily_h + np.random.normal(0, 3)

    temp = np.clip(temp, 2, 40)
    hum  = np.clip(hum, 25, 98)

    return pd.DataFrame({
        'datetime': dates,
        'outdoor_temp': np.round(temp, 1),
        'outdoor_humidity': np.round(hum, 1),
    })


# ============================================================
# 三、课表特征生成
# ============================================================
def generate_schedule(dates):
    """
    生成课表特征：
      - 周一至周五上课(8-12, 14-18)，周末空置
      - 每节课有不同预期出勤人数
      - course_intensity: 课表使用强度指数 (0~1)
    """
    n = len(dates)
    is_weekday = (dates.dt.dayofweek < 5).astype(int).values
    hours = dates.dt.hour.values

    is_class = np.zeros(n, dtype=int)
    occupancy = np.zeros(n)
    intensity = np.zeros(n)

    for i in range(n):
        if is_weekday[i] == 0:
            continue
        h = hours[i]
        for idx, (s, e) in enumerate(CLASS_PERIODS):
            if s <= h < e:
                is_class[i] = 1
                # 同一节课人数一致（用日期+课节做种子）
                seed = dates[i].dayofyear * 10 + idx
                rng = np.random.RandomState(seed)
                occ = rng.randint(20, MAX_CAPACITY + 1)
                occupancy[i] = occ
                intensity[i] = occ / MAX_CAPACITY
                break

    return pd.DataFrame({
        'datetime': dates,
        'is_weekday': is_weekday,
        'is_class_time': is_class,
        'expected_occupancy': occupancy.astype(int),
        'course_intensity': np.round(intensity, 3),
    })


# ============================================================
# 四、室内环境 + 能耗仿真核心
# ============================================================
def simulate_environment(weather_df, schedule_df, mode='traditional'):
    """
    逐小时仿真室内温湿度、CO₂、空调能耗。

    mode:
      'traditional' —— 统一26°C设定，下课不关空调(空转)，不开窗通风
      'optimized'  —— 根据课表+CO₂+室外温湿度动态调节设定温度与通风
    """
    n = len(weather_df)
    indoor_t = np.zeros(n)
    indoor_h = np.zeros(n)
    co2 = np.zeros(n)
    energy = np.zeros(n)
    setpoint = np.zeros(n)
    win_open = np.zeros(n)
    ac_on = np.zeros(n)

    # 初始值
    indoor_t[0] = weather_df['outdoor_temp'].iloc[0]
    indoor_h[0] = weather_df['outdoor_humidity'].iloc[0]
    co2[0] = CO2_OUTDOOR

    # 热惯性 (kJ/K)
    thermal_mass = ROOM_VOLUME * AIR_DENSITY * AIR_CP  # ~195.7 kJ/K

    for i in range(1, n):
        out_t  = weather_df['outdoor_temp'].iloc[i]
        out_h  = weather_df['outdoor_humidity'].iloc[i]
        occ    = schedule_df['expected_occupancy'].iloc[i]
        is_cls = schedule_df['is_class_time'].iloc[i]
        p_t    = indoor_t[i-1]
        p_h    = indoor_h[i-1]
        p_co2  = co2[i-1]

        # ---- 决策逻辑 ----
        if mode == 'traditional':
            # 传统模式
            setpoint[i] = 26.0
            # 室外>20°C就开空调（包括无课空转）
            if out_t > 20:
                ac_on[i] = 1
                win_open[i] = 0
            else:
                ac_on[i] = 0
                win_open[i] = 0
        else:
            # 优化模式
            if is_cls == 0:
                # 无课 → 关空调
                ac_on[i] = 0
                setpoint[i] = 0
                # CO₂偏高或室外温度适宜 → 开窗
                if p_co2 > 800 or (18 <= out_t <= 28):
                    win_open[i] = 1
                else:
                    win_open[i] = 0
            else:
                # 有课
                if out_t > 28:
                    ac_on[i] = 1
                    setpoint[i] = 27.0
                    win_open[i] = 1 if p_co2 > 800 else 0
                elif out_t > 22:
                    ac_on[i] = 1
                    setpoint[i] = 28.0
                    win_open[i] = 1 if p_co2 > 700 else 0
                elif out_t >= 15:
                    # 温和：自然通风为主，CO₂高时补空调
                    ac_on[i] = 1 if p_co2 > 900 else 0
                    setpoint[i] = 28.0 if p_co2 > 900 else 0
                    win_open[i] = 1
                else:
                    # 室外偏凉，纯自然通风
                    ac_on[i] = 0
                    setpoint[i] = 0
                    win_open[i] = 1

        # ---- 换气次数 ----
        if win_open[i]:
            ach = CH_WINDOW
        elif ac_on[i]:
            ach = ACH_AC
        else:
            ach = ACH_CLOSED

        # ---- 热负荷计算 ----
        # 传导热
        q_cond = (U_WALL * WALL_AREA + U_ROOF * ROOF_AREA + U_WINDOW * WINDOW_AREA) \
                 * (out_t - p_t) / 1000  # kW

        # 太阳辐射得热（简化，白天有）
        h_of_day = weather_df['datetime'].iloc[i].hour
        solar = 2.0 * WINDOW_AREA * max(0, np.sin(np.pi * (h_of_day - 6) / 12)) / 1000  # kW
        if h_of_day < 6 or h_of_day > 18:
            solar = 0

        # 人员 + 设备热负荷
        q_people = occ * PEOPLE_HEAT
        q_equip  = EQUIP_HEAT_CLASS if is_cls else EQUIP_HEAT_IDLE

        # 通风热负荷
        q_vent = ach * ROOM_VOLUME * AIR_DENSITY * AIR_CP * (out_t - p_t) / 3600  # kW

        # 总显热负荷
        q_total = q_cond + solar + q_people + q_equip + q_vent

        # ---- 空调能耗 ----
        if ac_on[i] and setpoint[i] > 0 and setpoint[i] < p_t:
            # 需要制冷
            cooling = min(q_total, AC_MAX_COOLING)
            energy[i] = cooling / AC_COP
            net_q = q_total - cooling
        elif ac_on[i]:
            # 空调开但无需制冷（传统模式空转）
            energy[i] = AC_STANDBY_POWER
            net_q = q_total
        else:
            energy[i] = 0
            net_q = q_total

        # 添加随机噪声（模拟真实波动：传感器误差、人员活动随机性等）
        if energy[i] > 0:
            energy[i] *= (1 + np.random.normal(0, 0.08))  # 8%相对噪声
            energy[i] = max(0, energy[i])

        # ---- 室内温度更新 ----
        dt_temp = net_q * 3600 / (thermal_mass * 1000)  # °C/h (注意单位)
        # 实际中热惯性更大，缩小变化率
        new_t = p_t + dt_temp * 0.3
        # 空调控温拉回设定值
        if ac_on[i] and setpoint[i] > 0:
            new_t = new_t + 0.4 * (setpoint[i] - new_t)
            new_t = np.clip(new_t, setpoint[i] - 1.5, setpoint[i] + 1.5)
        # 限制漂移到室外温度（极端情况）
        if not ac_on[i]:
            new_t = new_t + 0.1 * (out_t - new_t)

        indoor_t[i] = new_t

        # ---- 室内湿度更新 ----
        if ac_on[i]:
            new_h = p_h - 1.5 + 0.3 * (out_h - p_h) / 10
        elif win_open[i]:
            new_h = p_h + 0.6 * (out_h - p_h)
        else:
            new_h = p_h + 0.1 * (out_h - p_h)
        indoor_h[i] = np.clip(new_h, 25, 95)

        # ---- CO₂更新 ----
        # dC/dt = (G·N / V) - ACH·(C - C_out)   [ppm/h]
        co2_gen = (CO2_PER_PERSON_LPS * 3.6 * occ) / ROOM_VOLUME * 1e6  # ppm/h
        co2_vent = ach * (p_co2 - CO2_OUTDOOR)
        new_co2 = p_co2 + co2_gen - co2_vent
        co2[i] = np.clip(new_co2, CO2_OUTDOOR, 5000)

    result = pd.DataFrame({
        'datetime': weather_df['datetime'],
        'outdoor_temp': weather_df['outdoor_temp'],
        'outdoor_humidity': weather_df['outdoor_humidity'],
        'indoor_temp': np.round(indoor_t, 1),
        'indoor_humidity': np.round(indoor_h, 1),
        'co2': np.round(co2, 0),
        'occupancy': schedule_df['expected_occupancy'],
        'is_class_time': schedule_df['is_class_time'],
        'course_intensity': schedule_df['course_intensity'],
        'ac_setpoint': np.round(setpoint, 1),
        'ac_on': ac_on.astype(int),
        'window_open': win_open.astype(int),
        'ac_energy': np.round(energy, 3),
    })
    return result


# ============================================================
# 五、主函数
# ============================================================
def main():
    np.random.seed(42)
    print("=" * 60)
    print("  高校教室能耗仿真数据集生成")
    print("  地区: 桂林  |  时间范围: 2024全年  |  频率: 小时级")
    print("=" * 60)

    # 路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1) 气象数据
    print("\n[1/3] 生成桂林气象数据...")
    weather = generate_weather(START_DATE, END_DATE)
    print(f"  气象数据: {len(weather)} 行, {weather['datetime'].iloc[0]} ~ {weather['datetime'].iloc[-1]}")

    # 2) 课表特征
    print("[2/3] 生成课表特征...")
    schedule = generate_schedule(weather['datetime'])
    print(f"  课表数据: {len(schedule)} 行")
    print(f"  上课时段占比: {schedule['is_class_time'].mean()*100:.1f}%")

    # 3) 仿真室内环境（传统模式作为主数据集）
    print("[3/3] 仿真室内环境与能耗（传统模式）...")
    dataset = simulate_environment(weather, schedule, mode='traditional')

    # 同时生成优化模式数据（用于仿真对比实验）
    print("       仿真室内环境与能耗（优化模式）...")
    dataset_opt = simulate_environment(weather, schedule, mode='optimized')

    # 保存
    main_path = os.path.join(data_dir, 'classroom_energy_dataset.csv')
    opt_path  = os.path.join(data_dir, 'classroom_energy_optimized.csv')
    dataset.to_csv(main_path, index=False, encoding='utf-8-sig')
    dataset_opt.to_csv(opt_path, index=False, encoding='utf-8-sig')

    print(f"\n传统模式数据已保存: {main_path}")
    print(f"优化模式数据已保存: {opt_path}")
    print(f"\n数据集特征列: {list(dataset.columns)}")
    print(f"总行数: {len(dataset)}")
    print(f"时间范围: {dataset['datetime'].iloc[0]} ~ {dataset['datetime'].iloc[-1]}")

    # 统计摘要
    print("\n=== 数据统计摘要 ===")
    print(f"室外温度范围: {dataset['outdoor_temp'].min():.1f} ~ {dataset['outdoor_temp'].max():.1f} °C")
    print(f"室内温度范围: {dataset['indoor_temp'].min():.1f} ~ {dataset['indoor_temp'].max():.1f} °C")
    print(f"CO₂范围: {dataset['co2'].min():.0f} ~ {dataset['co2'].max():.0f} ppm")
    print(f"传统模式总能耗: {dataset['ac_energy'].sum():.1f} kWh")
    print(f"优化模式总能耗: {dataset_opt['ac_energy'].sum():.1f} kWh")
    savings = (1 - dataset_opt['ac_energy'].sum() / dataset['ac_energy'].sum()) * 100
    print(f"理论节能率: {savings:.1f}%")

    print("\n数据生成完成！")


if __name__ == '__main__':
    main()
