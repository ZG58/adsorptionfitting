
from ._settings import settings
import numpy as np
import inspect
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import stats
from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class FitParameter:
    """
    存储拟合参数
    """
    name: str
    value: float = 1.
    sigma: float = 0.
    fixed: bool = False

@dataclass
class FitModel:
    """
    存储模型
    """
    func: Any
    jac: Any
    weight: str
    fitpars: List[FitParameter]
    description: str = ''

    def evaluate(self, x):
        return self.func(x, *(par.value for par in self.fitpars))

    def get_numfitpars(self):
        return sum([not par.fixed for par in self.fitpars])


@dataclass
class FitData:
    x: np.array  # x 数据
    y: np.array  # y 数据
    xe: np.array = None  # x 值的误差数据
    ye: np.array = None  # y 值的误差数据
    mask: List[bool] = field(init=False)  # 掩码，用于筛选数据点

    def __post_init__(self):
        self.set_mask(-np.inf, np.inf)  # 初始化时设置掩码，包含所有数据点

    def get(self):
        result = (var[self.mask] if var is not None else None for var in [self.x, self.y, self.xe, self.ye])
        return result  # 返回筛选后的 x、y 及其误差数据

    def set_mask(self, xmin, xmax):
        self.mask = [xmin <= x <= xmax for x in self.x]  # 根据给定的最小和最大值设置掩码

    def get_numfitpoints(self):
        return len(self.x[self.mask])  # 返回有效数据点的数量


class Fitter:
    """ 处理拟合的类 """

    WEIGHTOPTIONS = ('none', 'relative', 'absolute')  # 权重选项
    MODELS = (
        'Langmuir',  # Langmuir 吸附模型
        'Freundlich',  # Freundlich 吸附模型
        'BET',  # BET 吸附模型
        'Temkin',  # Temkin 吸附模型
        'Dubinin-Radushkevich',  # Dubinin-Radushkevich 吸附模型
        'Toth',  # Toth 吸附模型
        'Dual-site',  # 双位点模型
        'Sips'  # Sips 吸附模型
    )

    def __init__(self, func, xdata, ydata, xerr, yerr, p0, absolute_sigma, jac, **kwargs):
        self.kwargs = kwargs  # 其他关键字参数
        self.data = self._init_data(xdata, ydata, xerr, yerr)  # 初始化数据
        self.model = self._init_model(func, p0, absolute_sigma, jac)  # 初始化模型
        self.fit_is_valid = False  # 当计算出有效拟合时设置为 True
        self.mean_squared_error = None  # 均方误差
        self.fitreport = {}  # 拟合报告

    def _init_data(self, x, y, xe, ye):
        # 验证数据
        for var in [x, y]:
            if type(var) is not np.ndarray:
                raise Exception('数据应为 numpy 数组类型')
        if len(x) != len(y):
            raise Exception('xdata 和 ydata 应具有相同的长度')

        # 获取误差数据（如果提供）
        if ye is not None:
            if type(ye) is not np.ndarray:
                raise Exception('数据应为 numpy 数组类型')
            if len(ye) != len(y):
                raise Exception('yerr 和 ydata 应具有相同的长度')

        if xe is not None:
            if type(xe) is not np.ndarray:
                raise Exception('数据应为 numpy 数组类型')
            if len(xe) != len(x):
                raise Exception('xerr 和 xdata 应具有相同的长度')

        return FitData(x, y, xe, ye)  # 返回初始化后的 FitData 实例

    def _init_model(self, func, p0, absolute_sigma, jac):
        # 验证函数
        if not callable(func):
            raise Exception('不是有效的拟合函数')

        # 从函数中确定拟合参数
        __args = inspect.signature(func).parameters
        args = [arg.name for arg in __args.values()]  # 获取参数名称

        # 如果未指定初始值，则将初始值设置为 1
        if p0 is None:
            p0 = [1.] * len(args[1:])  # 默认初始值为 1

        # 创建拟合参数
        fitpars = [FitParameter(arg, value) for arg, value in zip(args[1:], p0)]

        # 进行其他修改
        if self.data.ye is not None:
            if absolute_sigma:
                weight = self.WEIGHTOPTIONS[2]  # 绝对权重
            else:
                weight = self.WEIGHTOPTIONS[1]  # 相对权重
        else:
            weight = self.WEIGHTOPTIONS[0]  # 无权重

        # 创建并返回 FitModel 类
        if func.__doc__ is None:
            description = '没有模型信息'
        else:
            description = strip_leading_spaces(func.__doc__)  # 获取模型描述

        afitmodel = FitModel(func, jac, weight, fitpars, description)
        return afitmodel  # 返回拟合模型实例

    def fit(self):
        """
        执行拟合
        """
        # 准备模型和数据
        p0 = [fitpar.value for fitpar in self.model.fitpars]  # 初始拟合参数值
        pF = [fitpar.fixed for fitpar in self.model.fitpars]  # 拟合参数是否固定
        x, y, xe, ye = self.data.get()  # 获取数据

        # 检查自由拟合参数的数量
        if self.model.get_numfitpars() == 0:
            raise OptimizeWarning('至少应该有一个自由拟合参数')

        if self._degrees_of_freedom() <= 0:
            raise OptimizeWarning("自由度（dof）至少应该为 1。" +
                                  " 尝试增加数据点的数量或减少自由拟合参数的数量。")

        absolute_sigma = self.model.weight == self.WEIGHTOPTIONS[2]  # 检查是否为绝对权重
        if self.model.weight == self.WEIGHTOPTIONS[0]:
            ye = np.ones(len(y))  # 误差为 1 相当于没有权重

        # 调用拟合函数
        popt, pcov = curve_fit_wrapper(
            self.model.func, x, y, sigma=ye, p0=p0, pF=pF,
            absolute_sigma=absolute_sigma, jac=self.model.jac
        )

        # 处理结果
        self.fit_is_valid = True  # 标记拟合为有效
        stderrors = np.sqrt(np.diag(pcov))  # 计算标准误差
        for fitpar, value, stderr in zip(self.model.fitpars, popt, stderrors):
            fitpar.value = value  # 更新拟合参数的值
            fitpar.sigma = stderr  # 更新拟合参数的标准误差

        self.mean_squared_error = sum(((y - self.model.evaluate(x)) / ye) ** 2)  # 计算均方误差

        self._create_report()  # 创建拟合报告
        return popt, pcov  # 返回拟合结果和协方差矩阵

    def get_curve(self, xmin=None, xmax=None, numpoints=settings['MODEL_NUMPOINTS']):
        if xmin is None: xmin = self.data.x.min()  # 设置 x 的最小值
        if xmax is None: xmax = self.data.x.max()  # 设置 x 的最大值
        xcurve = np.linspace(xmin, xmax, numpoints)  # 生成曲线上的 x 值
        ycurve = self.model.evaluate(xcurve)  # 计算对应的 y 值
        return (xcurve, ycurve)  # 返回 x 和 y 的曲线数据

    def get_fitcurve(self, xmin=None, xmax=None, numpoints=settings['MODEL_NUMPOINTS']):
        if not self.fit_is_valid:
            return None  # 如果拟合无效，返回 None
        return self.get_curve(xmin, xmax, numpoints)  # 返回拟合曲线

    def get_residuals(self, check=True):
        """
        返回残差，即 y - f(x)
        如果 check 为 True（默认值），则仅在执行有效拟合时返回值。
        """
        if not self.fit_is_valid and check:  # 如果拟合无效并且需要检查
            return None  # 返回 None
        return self.data.y - self.model.evaluate(self.data.x)  # 计算并返回残差

    def _degrees_of_freedom(self):
        return int(self.data.get_numfitpoints() - self.model.get_numfitpars())  # 计算自由度

    def _create_report(self):
        def pars_to_dict():
            # 将拟合参数转换为字典
            parsdict = {par.name: dict(value=par.value, stderr=par.sigma, fixed=par.fixed) for par in
                        self.model.fitpars}
            return parsdict  # 返回参数字典

        # 创建拟合报告
        self.fitreport = {
            'FITPARAMETERS': {
                'model': self.model.description,  # 模型描述
                'weight': self.model.weight,  # 权重
                'N': self.data.get_numfitpoints(),  # 数据点数量
                'dof': self._degrees_of_freedom(),  # 自由度
                't95-val': stats.t.ppf(0.975, self._degrees_of_freedom())  # 95% 置信区间的 t 值
            },
            'FITRESULTS': pars_to_dict(),  # 拟合结果
            'STATISTICS': {
                'Smin': self.mean_squared_error  # 最小均方误差
            }
        }

    def get_report(self):
        return self.fitreport  # 返回拟合报告

    def get_weightoptions(self):
        if self.data.ye is not None:  # 如果存在 y 误差数据
            return self.WEIGHTOPTIONS  # 返回所有权重选项
        else:
            # 如果没有误差数据，则仅返回一个选项
            return self.WEIGHTOPTIONS[0:1]  # 只返回无权重选项

    def get_models(self):
        return self.MODELS


def curve_fit_wrapper(func, *pargs, p0=None, pF=None, jac=None, **kwargs):
    """
    对 scipy curve_fit() 函数的封装，以允许固定参数
    调用签名与 curve_fit() 函数相同，除了：
    pF : 1D numpy 数组，大小为 n，n 是函数的拟合参数数量
    返回 popt 和 cov 矩阵，方式与原始 curve_fit() 函数相同
    """

    # 提取函数 func 的参数
    __args = inspect.signature(func).parameters
    args = [arg.name for arg in __args.values()]

    # 如果未在 kwargs 中提供，则将 pF 和 p0 设置为默认值
    if pF is None:
        pF = np.array([False for _ in args[1:]])  # 将所有参数设置为可自由拟合
    if p0 is None:
        p0 = np.array([1 for _ in args[1:]])  # 将所有初始值设置为 1

    # 构造新的函数参数列表和传递给原始函数的函数参数列表
    newfunc_args = [args[0]] + [arg for arg, fix in zip(args[1:], pF) if not fix]
    orifunc_args = [args[0]] + [arg if not fix else str(p) for arg, fix, p in zip(args[1:], pF, p0)]

    # 定义新的函数作为 lambda 表达式，并求值为函数
    fit_func = eval(f"lambda {', '.join(newfunc_args)} : func({', '.join(orifunc_args)})", locals())

    # 定义新的雅可比函数（如果指定）作为 lambda 表达式，并求值为函数
    if callable(jac):
        indices = np.array([index for index, value in enumerate(pF) if value == False])
        fit_jac = eval(f"lambda {', '.join(newfunc_args)} : jac({', '.join(orifunc_args)})[:, indices]", locals())
    else:
        fit_jac = jac

    # 为自由拟合参数填充初始值列表
    p0_fit = np.array([p for p, fix in zip(p0, pF) if not fix])

    # 使用简化的函数进行拟合
    popt, cov = curve_fit(fit_func, *pargs, p0=p0_fit, jac=fit_jac, **kwargs)

    # 重建 popt 和 cov，以包含固定参数
    p0_fix = [p for p, fix in zip(p0, pF) if fix]  # 固定参数的值
    id_fix = np.where(pF)[0]  # 固定参数的索引
    for id, p in zip(id_fix, p0_fix):
        popt = np.insert(popt, id, p, axis=0)  # 在自由拟合参数处填充 popt

    # 重建协方差矩阵，以包含固定和优化的参数
    for id in id_fix:
        cov = np.insert(cov, id, 0, axis=1)  # 为固定参数添加零行和列
        cov = np.insert(cov, id, 0, axis=0)

    return popt, cov  # 返回优化后的参数和协方差矩阵



def value_to_string(name, value, error, fixed):
    """
    将值、误差和是否固定的状态转换为字符串表示
    """

    def get_exponent(value):
        """ 返回通过 :.e 格式说明符生成的指数作为整数 """
        deci = 5
        s = f'{value:.{deci}e}'
        index_sign = s.find('e') + 1
        return int(s[index_sign:])

    def float_to_string(value, exponent, sig_digits=2):
        """
        返回数字的科学记数法字符串表示
        value: 被转换的数字
        exponent: 用于表示的指数
        sig_digits: 使用的有效数字数量
        """
        deci = sig_digits + exponent - get_exponent(value) - 1
        if deci < 0:
            deci = 0
        result1 = f'{value / 10 ** exponent:.{deci}f}'
        result2 = f'{10 ** exponent:.0e}'[1:]
        return result1, result2

    def to_latex(value_str, exponent, error_str=None):
        """ 返回 latex 字符串
        (value_str +/- error_str) x 10^(exponent) """
        if error_str:
            latex_string = '$= (' + value_str + '\pm' + error_str +  r')\times$' + f'$10^{{{exponent}}}$'
        else:
            latex_string = '$=' + value_str + r'\times$' + f'$10^{{{exponent}}}$'
        return latex_string

    x_e = get_exponent(value)  # 获取值的指数
    if fixed:
        value_str, _ = float_to_string(value, x_e, settings['CM_SIG_DIGITS_NO_ERROR'])
        error_str = None
        exponent = x_e
    else:
        dx_e = get_exponent(error)  # 获取误差的指数
        if x_e >= dx_e:
            value_str, _ = float_to_string(value, x_e, settings['CM_SIG_DIGITS'] + x_e - dx_e)
            error_str, _ = float_to_string(error, x_e, settings['CM_SIG_DIGITS'])
            exponent = x_e
        else:
            value_str, _ = float_to_string(value, dx_e, settings['CM_SIG_DIGITS'] + x_e - dx_e)
            error_str, _ = float_to_string(error, dx_e, settings['CM_SIG_DIGITS'])
            exponent = dx_e
    combined = name + to_latex(value_str, exponent, error_str)  # 合并名称和latex字符串

    return combined


def float_to_str(value, digits):
    """
    返回值的字符串表示，以科学记数法表示，指定有效数字的数量为 digits
    """
    return f'{value:1.{digits}e}'


def strip_leading_spaces(text):
    """ 移除文本中的前导空格 """
    while text.count('\n    ') > 0:
        text = text.replace('\n    ', '\n')
    return text
