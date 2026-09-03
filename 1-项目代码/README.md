# 基于AI的高校教室能耗预测与智能节能决策系统

> 大创项目 | 桂林 | 多源数据融合AI预测 + 能耗与室内空气质量协同优化 + 轻量化无硬件改造

## 项目简介

本项目面向高校教室场景，利用AI技术预测空调能耗，并基于双约束决策算法（能耗最小化 + CO₂/温度达标）输出智能节能调控建议。系统支持**实时手动输入**和**CSV批量导入**两种模式，可直接用于教室空调管理。

## 快速开始（一键启动）

### 方式一：双击启动（推荐）

双击项目根目录下的 `启动系统.bat` 文件，系统会自动完成：
1. 检查Python环境
2. 自动安装缺失依赖
3. 生成仿真数据（首次运行）
4. 训练AI模型（首次运行）
5. 启动可视化界面

### 方式二：命令行启动

```bash
python run.py
```

系统启动后，浏览器打开 `http://localhost:8501` 即可使用。

## 项目结构

```
dachuang-project-code/
├── 启动系统.bat                        # Windows双击启动脚本
├── run.py                              # 一键启动（检查依赖+自动初始化+启动）
├── data/                               # 数据目录
│   ├── classroom_energy_dataset.csv        # 传统模式仿真数据集
│   ├── classroom_energy_optimized.csv      # 优化模式仿真数据集
│   ├── classroom_energy_ai_optimized.csv   # AI决策算法仿真数据
│   └── input_template.csv                  # 数据录入模板（供用户填写真实数据）
├── models/                            # 训练好的模型
│   ├── rf_model.pkl                        # RandomForest模型
│   ├── energy_model.pkl                    # 模型类型标记
│   ├── model_info.pkl                      # 模型配置信息
│   ├── feature_scaler.pkl                  # 特征归一化器
│   └── feature_cols.pkl                    # 特征列名
├── src/                                # 源代码
│   ├── data_generation.py                  # 仿真数据生成
│   ├── model_training.py                   # AI模型训练
│   ├── decision_algorithm.py               # 双约束决策算法
│   ├── simulation_experiment.py            # 仿真对比实验
│   └── app.py                              # Streamlit可视化系统
├── results/                           # 实验结果和图表
│   ├── experiment_report.txt               # 实验报告
│   ├── model_metrics.txt                   # 模型精度指标
│   ├── prediction_vs_actual.png            # 预测对比图
│   ├── energy_comparison_bar.png           # 能耗对比柱状图
│   ├── energy_timeseries.png               # 能耗时间序列对比
│   ├── co2_timeseries.png                  # CO₂时间序列对比
│   └── metrics_summary.csv                 # 指标汇总
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.8+
- 依赖库：pandas, numpy, matplotlib, scikit-learn, streamlit, joblib
- 可选：PyTorch（用于LSTM模型；未安装时自动回退到RandomForest）

## 功能页面

系统包含6个功能页面：

### 1. 🎯 实时调控建议（核心实用功能）
- **手动输入模式**：通过滑块输入当前教室环境参数（室外温度、室内温度、湿度、CO₂、人数、是否上课），点击"生成建议"获得AI决策
- **CSV批量模式**：上传一天的环境数据CSV，系统批量生成全天调控建议表，可导出CSV供后勤人员执行
- **模板下载**：提供标准数据录入模板，用户填写真实传感器数据后上传

### 2. 📊 总览仪表盘
- 核心指标卡片（总能耗、AI能耗、节能量、CO₂均值）
- 综合趋势图（能耗、温度、CO₂三合一）

### 3. 📈 能耗预测
- 基于历史数据的AI能耗预测曲线
- 模型精度指标展示

### 4. 🌡️ 室内环境
- 温度、湿度、CO₂、人数实时参数
- 24小时趋势图

### 5. 🤖 AI调控建议（历史数据）
- 基于历史数据集的AI决策展示
- 24小时决策序列可视化

### 6. ⚖️ 模式对比
- 传统模式 vs AI优化模式对比图
- 实验报告展示

## 使用"实时调控建议"功能

### 场景一：手动输入（即时决策）

1. 打开系统，选择左侧导航"🎯 实时调控建议"
2. 选择"✍️ 手动输入"
3. 用滑块设置当前教室环境参数：
   - 室外温度、室外湿度
   - 室内温度
   - CO₂浓度
   - 教室人数
   - 是否上课时段
4. 点击"🚀 生成调控建议"
5. 系统输出：空调设定温度、是否开窗通风、通风时长、预计节能效果、详细决策依据

### 场景二：批量处理（全天决策）

1. 下载模板文件 `input_template.csv`
2. 按模板格式填写真实教室环境数据（24小时或更多）
3. 在系统中选择"📁 上传CSV批量处理"
4. 上传CSV文件
5. 点击"🚀 生成全天调控建议"
6. 查看决策结果表，可导出CSV

## 手动运行各模块（高级）

如需单独运行各模块：

```bash
# 生成仿真数据集
python src/data_generation.py

# 训练AI预测模型
python src/model_training.py

# 验证决策算法
python src/decision_algorithm.py

# 仿真对比实验
python src/simulation_experiment.py

# 启动可视化系统
python -m streamlit run src/app.py
```

## 数据方案说明

| 数据源 | 方案 |
|--------|------|
| 气象数据 | Python模拟桂林气候特征（夏季28-35°C，冬季8-15°C） |
| 课表数据 | 模拟周一至周五上课(8-12,14-18)，周末空置 |
| 室内CO₂ | 基于人员数与通风量的物理公式模拟 |
| 空调能耗 | 基于热力学公式的简化模型 |
| 室内温湿度 | 基于空调状态、室外参数、人员热负荷模拟 |
| 实时数据 | 用户手动输入或上传CSV模板 |

## 核心创新

1. **多源数据融合AI预测**：课表+气象+CO₂+能耗多维特征输入LSTM/RandomForest
2. **能耗与空气质量协同优化**：双约束决策（能耗最小 + CO₂≤1000ppm + 温度24-28°C）
3. **轻量化无硬件改造**：纯软件方案，部署门槛低
4. **实时决策功能**：可直接用于教室空调管理，支持手动输入和批量CSV导入

## 约束标准

- CO₂ ≤ 1000 ppm（GB/T 18883-2022 室内空气质量标准）
- 室内温度 24~28°C（热舒适区间）
- 碳排放系数 0.5 kg CO₂/kWh
