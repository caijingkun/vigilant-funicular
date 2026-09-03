# -*- coding: utf-8 -*-
"""
一键启动脚本
=====================================================
项目: 基于AI的高校教室能耗预测与智能节能决策系统

功能:
  1. 检查Python依赖是否安装（pandas/numpy/sklearn/streamlit等）
  2. 检查数据文件是否存在（不存在则自动生成）
  3. 检查模型文件是否存在（不存在则自动训练）
  4. 生成数据录入模板（首次运行）
  5. 启动Streamlit可视化应用

使用方法:
  python run.py

或直接双击 启动系统.bat
=====================================================
"""

import os
import sys
import subprocess
import importlib

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 将src加入路径
sys.path.insert(0, SRC_DIR)

# 必需依赖列表
REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'streamlit': 'streamlit',
    'joblib': 'joblib',
}


def check_and_install_deps():
    """
    检查Python依赖是否已安装。
    缺少的包尝试自动安装。
    """
    print("\n" + "=" * 60)
    print("  [1/4] 检查Python环境依赖...")
    print("=" * 60)

    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} (未安装)")
            missing.append(pip_name)

    if missing:
        print(f"\n  缺少 {len(missing)} 个依赖包，尝试自动安装...")
        for pkg in missing:
            print(f"  正在安装 {pkg}...")
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', pkg, '-q']
                )
                print(f"  ✅ {pkg} 安装成功")
            except subprocess.CalledProcessError:
                print(f"  ❌ {pkg} 安装失败，请手动运行: pip install {pkg}")
                return False

    print("  ✅ 所有依赖检查通过")
    return True


def check_data_files():
    """
    检查数据文件是否存在，不存在则自动生成。
    """
    print("\n" + "=" * 60)
    print("  [2/4] 检查数据文件...")
    print("=" * 60)

    main_csv = os.path.join(DATA_DIR, 'classroom_energy_dataset.csv')
    opt_csv = os.path.join(DATA_DIR, 'classroom_energy_optimized.csv')

    if os.path.exists(main_csv) and os.path.exists(opt_csv):
        print(f"  ✅ 数据文件已存在: {os.path.basename(main_csv)}")
        # 验证文件大小
        size = os.path.getsize(main_csv)
        if size > 1000:
            print(f"     文件大小: {size/1024:.0f} KB")
            return True
        else:
            print(f"  ⚠️ 数据文件异常（大小仅{size}字节），重新生成...")

    print("  📊 正在生成仿真数据集（约需10-20秒）...")
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        from data_generation import main as gen_main
        gen_main()
        print("  ✅ 数据生成完成")
        return True
    except Exception as e:
        print(f"  ❌ 数据生成失败: {e}")
        return False


def check_model_files():
    """
    检查模型文件是否存在，不存在则自动训练。
    """
    print("\n" + "=" * 60)
    print("  [3/4] 检查模型文件...")
    print("=" * 60)

    rf_model = os.path.join(MODEL_DIR, 'rf_model.pkl')
    model_info = os.path.join(MODEL_DIR, 'model_info.pkl')

    if os.path.exists(rf_model) and os.path.exists(model_info):
        size = os.path.getsize(rf_model)
        if size > 1000:
            print(f"  ✅ 模型文件已存在: rf_model.pkl ({size/1024:.0f} KB)")
            return True
        else:
            print(f"  ⚠️ 模型文件异常，重新训练...")

    print("  🤖 正在训练AI预测模型（约需30-60秒）...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    try:
        from model_training import main as train_main
        train_main()
        print("  ✅ 模型训练完成")
        return True
    except Exception as e:
        print(f"  ❌ 模型训练失败: {e}")
        print("  ⚠️ 模型未训练，但决策算法仍可使用（不影响实时调控建议功能）")
        return True  # 不阻塞启动，决策算法不依赖模型


def create_input_template():
    """生成数据录入模板CSV"""
    template_path = os.path.join(DATA_DIR, 'input_template.csv')
    if os.path.exists(template_path):
        print("  ✅ 数据模板已存在: input_template.csv")
        return

    import pandas as pd
    import numpy as np

    # 生成一天的示例模板（24小时）
    dates = pd.date_range('2024-07-15 00:00:00', periods=24, freq='h')
    template_data = pd.DataFrame({
        'datetime': dates.strftime('%Y-%m-%d %H:%M:%S'),
        'outdoor_temp': [24, 23, 22, 22, 22, 23, 25, 27, 29, 31, 33, 34,
                         35, 35, 34, 33, 31, 29, 28, 27, 26, 25, 25, 24],
        'outdoor_humidity': [75, 76, 77, 78, 78, 77, 74, 70, 66, 62, 58, 55,
                             53, 52, 54, 57, 62, 66, 69, 71, 72, 73, 74, 75],
        'indoor_temp': [26, 26, 25, 25, 25, 26, 27, 28, 28, 29, 30, 30,
                        30, 30, 29, 28, 28, 27, 27, 27, 26, 26, 26, 26],
        'indoor_humidity': [65, 65, 66, 66, 66, 65, 64, 62, 60, 58, 56, 55,
                            54, 54, 55, 57, 59, 61, 62, 63, 64, 64, 65, 65],
        'co2': [450, 440, 430, 425, 420, 415, 500, 800, 950, 1100, 1200, 1250,
                1300, 1280, 1150, 900, 700, 650, 550, 500, 480, 470, 460, 455],
        'occupancy': [0, 0, 0, 0, 0, 0, 10, 45, 50, 50, 50, 50,
                      50, 45, 0, 0, 40, 45, 50, 50, 30, 10, 0, 0],
        'is_class_time': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                          0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    })

    os.makedirs(DATA_DIR, exist_ok=True)
    template_data.to_csv(template_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 数据模板已生成: {template_path}")


def check_simulation_results():
    """检查仿真实验结果是否存在，不存在则运行"""
    report_path = os.path.join(RESULT_DIR, 'experiment_report.txt')
    if os.path.exists(report_path):
        print("  ✅ 仿真实验结果已存在")
        return

    print("  📊 正在运行仿真对比实验...")
    os.makedirs(RESULT_DIR, exist_ok=True)
    try:
        from simulation_experiment import main as sim_main
        sim_main()
        print("  ✅ 仿真实验完成")
    except Exception as e:
        print(f"  ⚠️ 仿真实验跳过: {e}")
        print("  （不影响系统启动，可稍后手动运行 simulation_experiment.py）")


def start_streamlit():
    """启动Streamlit应用"""
    print("\n" + "=" * 60)
    print("  [4/4] 启动Streamlit可视化应用...")
    print("=" * 60)

    app_path = os.path.join(SRC_DIR, 'app.py')
    port = 8501

    # 尝试启动，如果端口被占用则换端口
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_path,
        '--server.port', str(port),
        '--server.headless', 'true',
    ]

    print(f"  启动命令: streamlit run src/app.py --server.port {port}")
    print(f"  访问地址: http://localhost:{port}")
    print("\n" + "=" * 60)
    print("  🌱 系统启动中，浏览器将自动打开...")
    print("  如需停止，请按 Ctrl+C")
    print("=" * 60 + "\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n  系统已停止。")
    except Exception as e:
        print(f"\n  ❌ 启动失败: {e}")
        print("  请尝试手动运行: streamlit run src/app.py")


def main():
    """主函数：一键启动流程"""
    print("\n" + "=" * 60)
    print("  🌱 基于AI的高校教室能耗预测与智能节能决策系统")
    print("  一键启动脚本")
    print("=" * 60)

    # 步骤1：检查依赖
    if not check_and_install_deps():
        print("\n❌ 依赖检查失败，请手动安装依赖后重试。")
        print("  命令: pip install -r requirements.txt")
        input("按回车键退出...")
        return

    # 步骤2：检查数据
    if not check_data_files():
        print("\n❌ 数据生成失败，请检查错误信息。")
        input("按回车键退出...")
        return

    # 步骤3：检查模型
    check_model_files()

    # 步骤3.5：生成模板
    print("\n  检查数据录入模板...")
    create_input_template()

    # 步骤3.6：检查仿真实验结果
    check_simulation_results()

    # 步骤4：启动Streamlit
    start_streamlit()


if __name__ == '__main__':
    main()
