# -*- coding: utf-8 -*-

import numpy as np
from curvefitgui._gui import execute_gui
from curvefitgui._model import langmuir, toth



def linear_fit_gui(xdata, ydata, xerr=None, yerr=None, xlabel='x-axis', ylabel='y-axis', showgui=True):
    """
    图形用户界面用于线性拟合

    参数:
    ----------
    xdata : 1-D numpy 数组
        数据的 x 坐标
    ydata : 1-D numpy 数组
        数据的 y 坐标
    yerr : 1-D numpy 数组, 可选 (默认:None)
        y 值的误差/不确定性，用于加权拟合，
        其中相对权重定义为 1/yerr**2
        （为了兼容，也可以使用关键字 sigma 来达到相同效果）
    xerr : 1-D numpy 数组, 可选 (默认:None)
        x 值的误差。仅用于绘制误差条，在拟合时忽略
    xlabel : 字符串, 可选 (默认:'x-values')
        图中 x 轴的标题
    ylabel : 字符串, 可选 (默认:'y-values')
        图中 y 轴的标题
    showgui : 布尔值, 可选 (默认=True)
        如果为 True，则显示 GUI，否则不显示

    返回值:
    --------
    popt : numpy 数组
        拟合参数的最优值
    pcov : 2D numpy 数组
        popt 的估计协方差矩阵


    注意:
    ------
    当前未实现 curve_fit() 的所有功能。求解器仅限于使用
    Levenberg-Marquardt 算法。

    示例:
    ---------

        # 定义 x 和 y 数据为长度相等的一维 numpy 数组
        xdata = np.array([1, 2, 3, 4, 5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])

        # 可选地定义 ydata 的误差
        yerr = np.array([0.5, 0.4, 0.6, 0.5, 0.8])

        # 可选地定义轴标题
        xlabel = 'time / s'
        ylabel = 'height / m'

        # 执行函数
        linear_fit_gui(xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)

    """

    # 创建拟合函数
    def f(x, a, b):
        """
        线性拟合
        y = ax + b
        a: 斜率
        b: 截距
        """
        return a * x + b

    p0 = None  # 初始参数
    absolute_sigma = False  # 是否使用绝对误差
    jac = None  # 雅可比矩阵
    kwargs = {}  # 其他参数

    # 调用 GUI 执行拟合
    res = execute_gui(f, xdata, ydata, xerr, yerr, p0, xlabel, ylabel,
                      absolute_sigma, jac, showgui, **kwargs)
    return res  # 返回拟合结果


def curve_fit_gui(f, xdata, ydata, xerr=None, yerr=None,
                  p0=None, xlabel='x-axis', ylabel='y-axis',
                  absolute_sigma=False, jac=None, showgui=True,
                  **kwargs):
    """
    图形用户界面用于 scipy 的 curve_fit() 函数。

    参数:
    ----------
    f : 可调用对象
        定义拟合函数的函数
    xdata : 1-D numpy 数组
        数据的 x 坐标
    ydata : 1-D numpy 数组
        数据的 y 坐标
    yerr : 1-D numpy 数组, 可选 (默认: None)
        y 值的误差/不确定性，用于加权拟合，
        其中相对权重定义为 1/yerr**2
        （为了兼容，也可以使用关键字 sigma 来达到相同效果）
    xerr : 1-D numpy 数组, 可选 (默认: None)
        x 值的误差。仅用于绘制误差条，在拟合时忽略
    xlabel : 字符串, 可选 (默认: 'x-values')
        图中 x 轴的标题
    ylabel : 字符串, 可选 (默认: 'y-values')
        图中 y 轴的标题
    p0 : 类似数组的对象, 可选
        拟合参数的初始值，如果未指定，则每个参数使用 1
    showgui : 布尔值, 可选 (默认: True)
        如果为 True，则显示 GUI，否则不显示
    absolute_sigma : 布尔值, 可选
        参见文档字符串 scipy.optimize.curve_fit()
    jac : 可调用对象, 可选
        参见文档字符串 scipy.optimize.curve_fit()
    kwargs :
        兼容性关键字参数（例如，可以使用 sigma 指定 y 的误差）


    返回值:
    --------
    popt : numpy 数组
        拟合参数的最优值
    pcov : 2D numpy 数组
        popt 的估计协方差矩阵

    另请参阅:
    ---------
    scipy.optimize.curve_fit()

    注意:
    ------
    当前未实现 curve_fit() 的所有功能。求解器仅限于使用
    Levenberg-Marquardt 算法。

    示例:
    ---------
    最小示例如下所示

        # 定义一个拟合函数
        def f(x, a, b):
            '''
            线性拟合
            函数: y = ax + b
            a: 斜率
            b: 截距
            '''
            return a * x + b

        # 定义 x 和 y 数据为长度相等的一维 numpy 数组
        xdata = np.array([1, 2, 3, 4, 5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])

        # 可选地定义 ydata 的误差
        yerr = np.array([0.5, 0.4, 0.6, 0.5, 0.8])

        # 可选地定义轴标题
        xlabel = 'time / s'
        ylabel = 'height / m'

        # 执行函数
        curve_fit_gui(f, xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)

    """
    # 'sigma' 和 'yerr' 两个关键字参数都可以用于指定 ydata 的误差
    # 如果指定了 'sigma'，则忽略 'yerr'。
    if 'sigma' in kwargs:
        yerr = kwargs['sigma']

    # 调用 GUI 执行拟合
    res = execute_gui(f, xdata, ydata, xerr, yerr, p0, xlabel, ylabel,
                      absolute_sigma, jac, showgui, **kwargs)
    return res  # 返回拟合结果


def __main__():

    # 定义 Langmuir 等温线函数
    # def langmuir(P, q_max, b):
    #     """
    #     Langmuir 等温线模型
    #     计算公式为 (q_max * b * P) / (1 + b * P)
    #
    #     参数:
    #     P : 浓度或压力值
    #     q_max : 最大吸附量，表示在表面上可以吸附的最大分子数量
    #     b : Langmuir 常数，表示吸附平衡常数，影响吸附能力
    #     """
    #     return (q_max * b * P) / (1 + b * P)
    #
    # def toth(P, q_max, b, n):
    #     """
    #     Toth 等温线模型
    #     计算公式为 (q_max * b * P) / ((1 + (b * P)^(1/n))^n)
    #
    #     参数:
    #     P : 浓度或压力值
    #     q_max : 最大吸附量，表示在表面上可以吸附的最大分子数量
    #     b : Toth 常数，表示吸附平衡常数，影响吸附能力
    #     n : 异质性指数，影响吸附表面的均匀性
    #     """
    #     return (q_max * b * P) / ((1 + (b * P) ** (1 / n)) ** n)

    # 创建测试数据
    P = np.linspace(0, 4, 50)  # 生成 0 到 4 的 50 个均匀分布的压力值
    q_max = 2.5  # 最大吸附量
    b = 1.3  # Langmuir 常数

    # 根据 Langmuir 等温线函数计算 y 值
    y = langmuir(P, q_max, b)

    # 生成随机噪声
    rng = np.random.default_rng()  # 创建随机数生成器
    yerr = 0.2 * np.ones_like(P)  # 设置 y 的误差为常数 0.2
    y_noise = yerr * rng.normal(size=P.size)  # 生成与 yerr 相同大小的随机噪声

    # 将噪声加到 y 值上，形成带噪声的数据
    ydata = y + y_noise

    # 执行 GUI 拟合
    popt, pcov = curve_fit_gui(toth, P, ydata, yerr=yerr, xlabel='x', ylabel='y', absolute_sigma =False)



if __name__ == "__main__":
    __main__()  # 运行主函数
