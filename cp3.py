import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 设置中文字体

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# Streamlit页面配置
st.title("临界阴极保护电位计算 Web 应用")
st.sidebar.header("输入参数")

# 添加下拉框供用户选择环焊缝类型
weld_type = st.sidebar.selectbox("请选择环焊缝类型", ["X70环焊缝", "X80环焊缝"])

# 添加下拉框供用户选择算法
algorithm = st.sidebar.selectbox("请选择预测算法", ["插值法", "XGBoost", "高斯回归"])

# 文件上传
uploaded_file = st.sidebar.file_uploader("请上传环焊缝在目标土壤中测得的消除IR降后的阴极极化曲线", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # 假设第一列是电流密度 i，第二列是电位 E
    i = data.iloc[:, 0].values.reshape(-1, 1)
    E = data.iloc[:, 1].values

    # 处理缺失值（删除包含 NaN 的行）
    valid_indices = ~np.isnan(i).flatten() & ~np.isnan(E)
    i = i[valid_indices].reshape(-1, 1)
    E = E[valid_indices]

    # 对 i 取对数变换
    i_log = np.log10(i)

    # 从数据集中随机选择 1/10 的点进行插值
    i_log_train = i_log.flatten()
    E_train = E.flatten()

    # 用户输入环境温度和土壤电阻率
    try:
        T = float(st.sidebar.text_input("请输入环境温度 (℃, 默认 25℃): ", value="25"))
        R = float(st.sidebar.text_input("请输入土壤电阻率 (Ω·m, 默认 100Ω·m): ", value="100"))
    except ValueError:
        st.sidebar.warning("输入无效，使用默认值 T=25℃ 和 R=100Ω·m")
        T = 25
        R = 100

    # 计算 FHmax 和 FHmin
    FHmax = -1.2
    if T > 60:
        FHmin = -0.95
    elif T < 40:
        if R < 1000:
            FHmin = -0.75
        else:
            FHmin = -0.65
    else:
        FHmin = -0.85

    # 根据选择的焊缝类型计算 i 的函数
    if weld_type == "X70环焊缝":
        def calculate_i_from_FH(FH):
            return (55.72 / (57.95 - FH) - 1) ** (1 / 1.37) * 0.11 / 1000
    else:  # X80环焊缝
        def calculate_i_from_FH(FH):
            return (64.32 / (63.41 - FH) - 1) ** (1 / 2.46) * 0.075 / 1000

    # 选择的算法处理
    if algorithm == "插值法":
        # 使用插值法进行预测
        interp_func = interp1d(i_log_train, E_train, bounds_error=False, fill_value="extrapolate")
        predict_func = lambda x: interp_func(np.log10(x))

    elif algorithm == "XGBoost":
        # 使用 XGBoost 进行预测
        model = XGBRegressor()
        model.fit(i_log, E)
        predict_func = lambda x: model.predict(np.log10(x).reshape(-1, 1))

    elif algorithm == "高斯回归":
        # 使用高斯过程回归进行预测
        kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        gp.fit(i_log, E)
        predict_func = lambda x: gp.predict(np.log10(x).reshape(-1, 1))

    # 计算临界负向电位
    if st.sidebar.button("计算临界负向电位"):
        FH_critical = 25
        i_critical = calculate_i_from_FH(FH_critical)

        try:
            E_critical = predict_func(i_critical)
            E_critical_CSE = E_critical + 0.075
            E_critical_CSE = max(E_critical_CSE, FHmax)  # 确保 E_critical_CSE 不超过 FHmax

            # 显示结果
            st.success(f"临界负向电位为：E = {E_critical_CSE:.2f} V")

        except Exception as e:
            st.sidebar.error(f"计算临界负向电位时出错: {e}")

    # 生成 FH 的范围
    FH_range = np.linspace(0.1, 50, 500)
    i_predicted = np.array([calculate_i_from_FH(FH) for FH in FH_range])

    # 过滤掉无效的 i_predicted 值
    valid_indices = ~np.isnan(i_predicted) & (i_predicted > 0)
    FH_range_valid = FH_range[valid_indices]
    i_predicted_valid = i_predicted[valid_indices]

    # 使用选定的算法预测 E
    try:
        E_predicted = predict_func(i_predicted_valid) + 0.075

        # 绘制图表
        plt.figure(figsize=(12, 8))
        plt.axvspan(0.1, 25, color='green', alpha=0.1)
        plt.text(12.5, np.min(E_predicted), '安全区', fontsize=16, va='center', ha='center', color='green', alpha=1)
        plt.axvspan(25, 35, color='yellow', alpha=0.1)
        plt.text(30, np.min(E_predicted), '氢脆风险区', fontsize=16, va='center', ha='center', color='orange', alpha=1)
        plt.axvspan(35, 50, color='red', alpha=0.1)
        plt.text(42.5, np.min(E_predicted), '氢脆断裂区', fontsize=16, va='center', ha='center', color='red', alpha=1)

        plt.plot(FH_range_valid, E_predicted, label='阴极保护电位 vs. 氢脆敏感性', color='blue')

        # 绘制 FHmax 和 FHmin 的虚线
        plt.axhline(y=FHmax, color='red', linestyle='--')
        plt.axhline(y=FHmin, color='red', linestyle='--')

        # 在图上标记 FH=25 的点
        plt.scatter([FH_critical], [E_critical_CSE], color='red', s=100, zorder=5)
        plt.text(FH_critical, E_critical_CSE, f'  (FH={FH_critical}%, E={E_critical_CSE:.2f} V)', fontsize=20,
                 va='bottom', ha='left', color='red')
        plt.xlim(0, 50)
        plt.xlabel('氢脆敏感性系数（%）')
        plt.ylabel('阴极保护电位 (V vs. CSE)')
        plt.title('阴极保护电位 vs. 氢脆敏感性系数')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 显示图表
        st.pyplot(plt)

    except Exception as e:
        st.error(f"请等待")
else:
    st.info("请上传一个 Excel 文件以开始。")
