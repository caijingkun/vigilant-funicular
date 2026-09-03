# -*- coding: utf-8 -*-
"""
双约束决策算法模块
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统

约束条件:
  ① 最小化空调能耗
  ② CO₂ ≤ 1000 ppm（GB/T 18883-2022 室内空气质量标准）
  ③ 室内温度 24~28°C（热舒适区间）

决策逻辑:
  - 无课时段 → 关闭空调（传统模式空转浪费）
  - 室外温湿度适宜且CO₂不超标 → 优先自然通风替代空调
  - CO₂接近上限 → 必须开窗通风，同时判断是否需要空调
  - 室外过热/过冷 → 空调为主，在约束下选能耗最低设定温度
  - 室内已在舒适区间 → 不开空调，微通风维持空气品质

输出:
  - 建议空调设定温度 (°C)
  - 是否建议开窗通风 (bool)
  - 建议通风时长 (分钟)
  - 决策依据说明 (str)
  - 预计节能效果 (kWh，相对26°C统一设定的传统模式)
=====================================================
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DecisionResult:
    """决策结果数据类"""
    ac_setpoint: float         # 建议空调设定温度 (°C)，0=关闭
    ac_on: bool                # 是否开启空调
    open_window: bool          # 是否建议开窗通风
    ventilation_minutes: int   # 建议通风时长 (分钟)
    reason: str                # 决策依据说明
    energy_saving: float       # 预计节能 (kWh，相对26°C基准)
    baseline_energy: float     # 传统模式基准能耗 (kWh)
    optimized_energy: float    # 优化模式预计能耗 (kWh)
    detail_reasons: list       # 逐条决策依据


class EnergyDecisionMaker:
    """
    双约束智能节能决策器
    输入预测能耗、环境参数、课表信息，输出最优空调控制策略。

    设计原则:
      - 输入参数自动校验，防止异常值导致决策失败
      - 综合考虑室外温湿度、室内温度、CO₂、人员数、课表等多维因素
      - 每条决策输出详细依据，便于运维人员理解
      - 量化节能效果，对比传统26°C统一设定
    """

    # 约束参数
    CO2_LIMIT = 1000           # CO₂上限 (GB/T 18883-2022)
    CO2_WARNING = 800          # CO₂预警阈值
    CO2_COMFORT = 600          # CO₂舒适阈值
    TEMP_MIN = 24              # 热舒适下限
    TEMP_MAX = 28              # 热舒适上限
    OUTDOOR_TEMP_MIN = 18      # 适宜通风的室外温度下限
    OUTDOOR_TEMP_MAX = 28      # 适宜通风的室外温度上限
    OUTDOOR_HUMIDITY_MAX = 80  # 适宜通风的室外湿度上限
    CARBON_FACTOR = 0.5        # 碳排放系数 kg CO₂/kWh

    def __init__(self, ac_power=5.0, ac_cop=3.2):
        """
        参数:
          ac_power: 空调总额定功率 (kW)
          ac_cop:   空调能效比
        """
        self.ac_power = ac_power
        self.ac_cop = ac_cop
        self.ac_max_cooling = ac_power * ac_cop

    def _validate_input(self, **kwargs):
        """
        输入参数校验，将异常值钳制到合理范围。
        防止传感器故障或人为输入错误导致决策异常。
        """
        # 温度类参数：钳制到 -20~50°C
        for key in ['outdoor_temp', 'indoor_temp']:
            if key in kwargs:
                kwargs[key] = float(np.clip(kwargs[key], -20, 50))

        # 湿度参数：钳制到 0~100%
        for key in ['outdoor_humidity', 'indoor_humidity']:
            if key in kwargs:
                kwargs[key] = float(np.clip(kwargs[key], 0, 100))

        # CO₂：钳制到 300~5000 ppm
        if 'current_co2' in kwargs:
            kwargs['current_co2'] = float(np.clip(kwargs['current_co2'], 300, 5000))

        # 人员数：钳制到 0~200
        if 'occupancy' in kwargs:
            kwargs['occupancy'] = int(max(0, min(200, kwargs['occupancy'])))

        # 布尔值强制转换
        if 'is_class_time' in kwargs:
            kwargs['is_class_time'] = bool(kwargs['is_class_time'])

        # 小时数
        if 'hour' in kwargs and kwargs['hour'] is not None:
            kwargs['hour'] = int(np.clip(kwargs['hour'], 0, 23))

        return kwargs

    def _estimate_baseline_energy(self, outdoor_temp, occupancy, is_class_time):
        """
        估算传统模式（26°C统一设定、无课空转）的基准能耗。
        用于计算AI决策的节能效果。

        返回: 基准能耗 (kWh/h)
        """
        if not is_class_time and outdoor_temp > 20:
            # 无课空转：待机功率
            return 0.3

        if outdoor_temp <= 20:
            # 室外低温，传统模式也不开空调
            return 0.0

        # 有课时：空调以26°C设定运行
        cooling = min(self._estimate_cooling(outdoor_temp, 26, occupancy),
                      self.ac_max_cooling)
        return cooling / self.ac_cop

    def _estimate_optimized_energy(self, outdoor_temp, setpoint, ac_on, occupancy):
        """估算优化模式的实际能耗"""
        if not ac_on or setpoint == 0:
            return 0.0
        cooling = min(self._estimate_cooling(outdoor_temp, setpoint, occupancy),
                      self.ac_max_cooling)
        return cooling / self.ac_cop

    def make_decision(self,
                      current_co2: float,
                      outdoor_temp: float,
                      outdoor_humidity: float,
                      indoor_temp: float,
                      occupancy: int,
                      is_class_time: bool,
                      hour: int = None) -> DecisionResult:
        """
        生成节能决策。

        参数:
          current_co2:       当前室内CO₂浓度 (ppm)
          outdoor_temp:      室外温度 (°C)
          outdoor_humidity:  室外湿度 (%)
          indoor_temp:       当前室内温度 (°C)
          occupancy:         当前/预期人员数
          is_class_time:     是否在上课时段
          hour:              当前小时 (0-23)

        返回:
          DecisionResult
        """
        # ======================================================
        # 输入校验
        # ======================================================
        kwargs = self._validate_input(
            current_co2=current_co2,
            outdoor_temp=outdoor_temp,
            outdoor_humidity=outdoor_humidity,
            indoor_temp=indoor_temp,
            occupancy=occupancy,
            is_class_time=is_class_time,
            hour=hour,
        )
        current_co2 = kwargs['current_co2']
        outdoor_temp = kwargs['outdoor_temp']
        outdoor_humidity = kwargs['outdoor_humidity']
        indoor_temp = kwargs['indoor_temp']
        occupancy = kwargs['occupancy']
        is_class_time = kwargs['is_class_time']
        hour = kwargs['hour'] if hour is not None else 12

        details = []

        # 计算基准能耗（传统26°C模式）
        baseline_energy = self._estimate_baseline_energy(
            outdoor_temp, occupancy, is_class_time)

        # ======================================================
        # 场景1: 无课时段 → 关闭空调
        # ======================================================
        if not is_class_time:
            # 无课时关闭空调（传统模式空转浪费）
            need_vent = (current_co2 > self.CO2_WARNING or
                         indoor_temp > self.TEMP_MAX)
            vent_min = 30 if need_vent else 0

            optimized_energy = 0.0
            saving = baseline_energy - optimized_energy

            details.append(f"当前为无课时段，传统模式空调空转浪费{baseline_energy:.2f}kWh")
            details.append(f"AI建议关闭空调，预计节能{saving:.2f}kWh")
            if need_vent:
                if current_co2 > self.CO2_WARNING:
                    details.append(f"CO₂={current_co2:.0f}ppm超过预警值({self.CO2_WARNING})，建议开窗通风30分钟换气")
                elif indoor_temp > self.TEMP_MAX:
                    details.append(f"室内温度={indoor_temp:.1f}°C超过舒适上限({self.TEMP_MAX}°C)，建议开窗降温")

            return DecisionResult(
                ac_setpoint=0,
                ac_on=False,
                open_window=need_vent,
                ventilation_minutes=vent_min,
                reason="无课时段：关闭空调以节能" +
                       ("；CO₂偏高，建议开窗通风30分钟" if need_vent else ""),
                energy_saving=saving,
                baseline_energy=baseline_energy,
                optimized_energy=optimized_energy,
                detail_reasons=details,
            )

        # ======================================================
        # 场景2: 有课时段 —— 判断室外条件是否适宜自然通风
        # ======================================================
        outdoor_suitable = (self.OUTDOOR_TEMP_MIN <= outdoor_temp <= self.OUTDOOR_TEMP_MAX and
                            outdoor_humidity <= self.OUTDOOR_HUMIDITY_MAX)
        co2_near_limit = current_co2 >= self.CO2_WARNING
        co2_safe = current_co2 < self.CO2_LIMIT

        # 场景2a: 室外温湿度适宜 + CO₂不超标 → 纯自然通风
        if outdoor_suitable and co2_safe:
            vent_min = 15 if occupancy > 30 else 10
            optimized_energy = 0.0
            saving = baseline_energy - optimized_energy

            details.append(f"室外温度={outdoor_temp:.0f}°C在适宜区间({self.OUTDOOR_TEMP_MIN}-{self.OUTDOOR_TEMP_MAX}°C)，湿度={outdoor_humidity:.0f}%≤{self.OUTDOOR_HUMIDITY_MAX}%")
            details.append(f"CO₂={current_co2:.0f}ppm低于标准限值({self.CO2_LIMIT})，空气质量达标")
            details.append(f"采用自然通风替代空调，通风{vent_min}分钟，预计节能{saving:.2f}kWh")

            return DecisionResult(
                ac_setpoint=0,
                ac_on=False,
                open_window=True,
                ventilation_minutes=vent_min,
                reason=f"室外条件适宜({outdoor_temp:.0f}°C,{outdoor_humidity:.0f}%)，"
                       f"采用自然通风替代空调，兼顾通风与节能",
                energy_saving=saving,
                baseline_energy=baseline_energy,
                optimized_energy=optimized_energy,
                detail_reasons=details,
            )

        # ======================================================
        # 场景3: CO₂接近上限 → 必须通风，同时决定是否开空调
        # ======================================================
        if co2_near_limit:
            open_win = True
            vent_min = 20

            details.append(f"CO₂={current_co2:.0f}ppm接近或超过预警值({self.CO2_WARNING})，必须开窗通风20分钟")

            if outdoor_temp > self.OUTDOOR_TEMP_MAX:
                # 室外过热：开窗引入热量，但仍需通风降CO₂
                # 空调设较高温度（28°C）以节能
                setpoint = self.TEMP_MAX
                cooling = min(self._estimate_cooling(outdoor_temp, setpoint, occupancy),
                              self.ac_max_cooling)
                optimized_energy = cooling / self.ac_cop
                saving = max(0, baseline_energy - optimized_energy)

                details.append(f"室外高温={outdoor_temp:.0f}°C超过{self.OUTDOOR_TEMP_MAX}°C，开窗会引入热量")
                details.append(f"空调设{setpoint}°C（约束上限）以最小化制冷负荷")
                details.append(f"传统模式26°C能耗={baseline_energy:.2f}kWh，优化模式能耗={optimized_energy:.2f}kWh，节能{saving:.2f}kWh")

                return DecisionResult(
                    ac_setpoint=setpoint,
                    ac_on=True,
                    open_window=open_win,
                    ventilation_minutes=vent_min,
                    reason=f"CO₂={current_co2:.0f}ppm接近上限({self.CO2_LIMIT})，"
                           f"开窗通风20分钟；室外高温设28°C运行空调",
                    energy_saving=saving,
                    baseline_energy=baseline_energy,
                    optimized_energy=optimized_energy,
                    detail_reasons=details,
                )
            elif outdoor_temp >= self.OUTDOOR_TEMP_MIN:
                # 室外不太热但CO₂高，通风为主
                optimized_energy = 0.0
                saving = baseline_energy - optimized_energy

                details.append(f"室外温度={outdoor_temp:.0f}°C在适宜范围，通风即可解决CO₂问题")
                details.append(f"不开空调，预计节能{saving:.2f}kWh")

                return DecisionResult(
                    ac_setpoint=0,
                    ac_on=False,
                    open_window=True,
                    ventilation_minutes=vent_min,
                    reason=f"CO₂={current_co2:.0f}ppm偏高，开窗通风20分钟；"
                           f"室外温度适宜不开空调",
                    energy_saving=saving,
                    baseline_energy=baseline_energy,
                    optimized_energy=optimized_energy,
                    detail_reasons=details,
                )
            else:
                # 室外过冷，微开窗+空调制热模式（简化为空调设24°C）
                setpoint = self.TEMP_MIN
                cooling = min(self._estimate_cooling(outdoor_temp, setpoint, occupancy),
                              self.ac_max_cooling)
                optimized_energy = cooling / self.ac_cop
                saving = max(0, baseline_energy - optimized_energy)

                details.append(f"室外低温={outdoor_temp:.0f}°C低于{self.OUTDOOR_TEMP_MIN}°C，需空调辅助")
                details.append(f"微开窗15分钟降CO₂+空调设{setpoint}°C维持热舒适")

                return DecisionResult(
                    ac_setpoint=setpoint,
                    ac_on=True,
                    open_window=True,
                    ventilation_minutes=15,
                    reason=f"CO₂={current_co2:.0f}ppm偏高且室外偏冷({outdoor_temp:.0f}°C)，"
                           f"微开窗15分钟+空调设24°C",
                    energy_saving=saving,
                    baseline_energy=baseline_energy,
                    optimized_energy=optimized_energy,
                    detail_reasons=details,
                )

        # ======================================================
        # 场景4: 室外过热/过冷，CO₂未超标 → 空调为主
        # ======================================================
        if outdoor_temp > self.OUTDOOR_TEMP_MAX:
            # 室外过热：选最高设定温度（28°C）以最小化能耗
            setpoint = self.TEMP_MAX
            cooling = min(self._estimate_cooling(outdoor_temp, setpoint, occupancy),
                          self.ac_max_cooling)
            optimized_energy = cooling / self.ac_cop
            saving = max(0, baseline_energy - optimized_energy)

            details.append(f"室外高温={outdoor_temp:.0f}°C超过{self.OUTDOOR_TEMP_MAX}°C，需空调制冷")
            details.append(f"设定温度{setpoint}°C（舒适区间上限），比传统26°C高{setpoint-26:.0f}°C，缩小室内外温差以减少制冷负荷")
            details.append(f"CO₂={current_co2:.0f}ppm正常，无需额外通风")
            details.append(f"传统模式26°C能耗={baseline_energy:.2f}kWh，优化模式能耗={optimized_energy:.2f}kWh，节能{saving:.2f}kWh")

            return DecisionResult(
                ac_setpoint=setpoint,
                ac_on=True,
                open_window=False,
                ventilation_minutes=0,
                reason=f"室外高温({outdoor_temp:.0f}°C)，空调设28°C（约束上限）"
                       f"以最小化能耗",
                energy_saving=saving,
                baseline_energy=baseline_energy,
                optimized_energy=optimized_energy,
                detail_reasons=details,
            )

        elif outdoor_temp < self.OUTDOOR_TEMP_MIN:
            # 室外过冷：选最低设定温度（24°C）
            setpoint = self.TEMP_MIN
            optimized_energy = 0.0  # 室外偏冷，室内通常不会过热，不需要制冷
            saving = baseline_energy - optimized_energy

            details.append(f"室外低温={outdoor_temp:.0f}°C低于{self.OUTDOOR_TEMP_MIN}°C")
            details.append(f"CO₂={current_co2:.0f}ppm正常，无需通风")
            details.append(f"室内温度{indoor_temp:.1f}°C，无需空调制冷，预计节能{saving:.2f}kWh")

            return DecisionResult(
                ac_setpoint=setpoint,
                ac_on=True,
                open_window=False,
                ventilation_minutes=0,
                reason=f"室外低温({outdoor_temp:.0f}°C)，空调设24°C维持热舒适",
                energy_saving=saving,
                baseline_energy=baseline_energy,
                optimized_energy=optimized_energy,
                detail_reasons=details,
            )

        else:
            # 中间地带：室外温度在18-28之间，CO₂未超标
            # 判断室内温度是否在舒适区间
            if self.TEMP_MIN <= indoor_temp <= self.TEMP_MAX:
                # 已在舒适区间，不开空调
                vent_min = 10 if occupancy > 30 else 5
                optimized_energy = 0.0
                saving = baseline_energy - optimized_energy

                details.append(f"室内温度={indoor_temp:.1f}°C在舒适区间({self.TEMP_MIN}-{self.TEMP_MAX}°C)")
                details.append(f"CO₂={current_co2:.0f}ppm正常，空气质量达标")
                details.append(f"开窗通风{vent_min}分钟维持空气品质，不开空调")
                details.append(f"预计节能{saving:.2f}kWh")

                return DecisionResult(
                    ac_setpoint=0,
                    ac_on=False,
                    open_window=True,
                    ventilation_minutes=vent_min,
                    reason="室内温度已处于舒适区间(24-28°C)，CO₂正常，"
                           "开窗通风维持空气品质即可",
                    energy_saving=saving,
                    baseline_energy=baseline_energy,
                    optimized_energy=optimized_energy,
                    detail_reasons=details,
                )
            else:
                # 室内偏离舒适区间，微调
                setpoint = self.TEMP_MAX if indoor_temp > self.TEMP_MAX else self.TEMP_MIN
                cooling = min(self._estimate_cooling(outdoor_temp, setpoint, occupancy),
                              self.ac_max_cooling)
                optimized_energy = cooling / self.ac_cop
                saving = max(0, baseline_energy - optimized_energy)

                details.append(f"室内温度={indoor_temp:.1f}°C偏离舒适区间({self.TEMP_MIN}-{self.TEMP_MAX}°C)")
                details.append(f"空调设{setpoint}°C进行调节")
                details.append(f"CO₂={current_co2:.0f}ppm正常，无需额外通风")

                return DecisionResult(
                    ac_setpoint=setpoint,
                    ac_on=True,
                    open_window=False,
                    ventilation_minutes=0,
                    reason=f"室内温度={indoor_temp:.0f}°C偏离舒适区间，"
                           f"空调设{setpoint}°C调节",
                    energy_saving=saving,
                    baseline_energy=baseline_energy,
                    optimized_energy=optimized_energy,
                    detail_reasons=details,
                )

    def _estimate_cooling(self, outdoor_temp, setpoint, occupancy):
        """
        简估算制冷负荷 (kW)。
        基于温差、人员热负荷的简化模型。
        """
        # 传导热（简化，与data_generation一致: 0.29 kW/K）
        delta_t = max(0, outdoor_temp - setpoint)
        q_cond = 0.30 * delta_t  # kW (简化传热系数)
        # 人员热负荷
        q_people = occupancy * 0.10  # kW
        # 设备热负荷
        q_equip = 1.2  # kW
        return q_cond + q_people + q_equip

    def batch_decide(self, df):
        """
        批量决策：对数据集的每一行生成决策。
        df 需包含: ac_energy, co2, outdoor_temp, outdoor_humidity,
                   indoor_temp, occupancy, is_class_time, datetime
        返回 DataFrame 补充决策列
        """
        import pandas as pd
        results = []
        for _, row in df.iterrows():
            r = self.make_decision(
                current_co2=row['co2'],
                outdoor_temp=row['outdoor_temp'],
                outdoor_humidity=row['outdoor_humidity'],
                indoor_temp=row['indoor_temp'],
                occupancy=row.get('occupancy', 0),
                is_class_time=bool(row['is_class_time']),
            )
            results.append({
                '建议设定温度': r.ac_setpoint if r.ac_on else 0,
                '是否开空调': '是' if r.ac_on else '否',
                '是否开窗通风': '是' if r.open_window else '否',
                '通风时长(分钟)': r.ventilation_minutes,
                '预计节能(kWh)': round(r.energy_saving, 3),
                '传统模式能耗(kWh)': round(r.baseline_energy, 3),
                '优化模式能耗(kWh)': round(r.optimized_energy, 3),
                '决策依据': r.reason,
            })
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(results)], axis=1)


# ============================================================
# 主函数：快速验证
# ============================================================
if __name__ == '__main__':
    dm = EnergyDecisionMaker()

    # 测试场景
    scenarios = [
        ("无课时段",  dict(current_co2=600, outdoor_temp=30,
                          outdoor_humidity=70, indoor_temp=27,
                          occupancy=0, is_class_time=False)),
        ("上课+CO₂正常+室外适宜", dict(current_co2=700, outdoor_temp=24,
                          outdoor_humidity=60, indoor_temp=25,
                          occupancy=45, is_class_time=True)),
        ("上课+CO₂超标风险", dict(current_co2=950, outdoor_temp=32,
                          outdoor_humidity=75, indoor_temp=28,
                          occupancy=50, is_class_time=True)),
        ("上课+室外高温", dict(current_co2=600, outdoor_temp=35,
                          outdoor_humidity=70, indoor_temp=30,
                          occupancy=40, is_class_time=True)),
    ]

    print("=" * 60)
    print("  双约束决策算法验证")
    print("=" * 60)
    for name, params in scenarios:
        r = dm.make_decision(**params)
        print(f"\n【{name}】")
        print(f"  输入: CO₂={params['current_co2']}ppm, 室外={params['outdoor_temp']}°C, "
              f"室内={params['indoor_temp']}°C, 人数={params['occupancy']}")
        print(f"  → 空调设定: {'关闭' if not r.ac_on else str(r.ac_setpoint)+'°C'}")
        print(f"  → 开窗通风: {'是' if r.open_window else '否'}"
              f"{'，'+str(r.ventilation_minutes)+'分钟' if r.ventilation_minutes else ''}")
        print(f"  → 传统模式能耗: {r.baseline_energy:.2f} kWh")
        print(f"  → 优化模式能耗: {r.optimized_energy:.2f} kWh")
        print(f"  → 预计节能: {r.energy_saving:.2f} kWh ({r.energy_saving*0.5:.2f} kg CO₂)")
        print(f"  → 依据: {r.reason}")
        print(f"  → 详细依据:")
        for d in r.detail_reasons:
            print(f"      • {d}")
