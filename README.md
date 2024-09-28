# scipy 的 curve_fit() 函数的图形界面

![图形界面](https://github.com/moosepy/curvefitgui/raw/master/images/curvefitgui1.png)

`curvefitgui` 是一个用于 scipy.optimize 包中非线性拟合函数 [scipy.optimise.curve_fit API 参考](https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.curve_fit.html?highlight=scipy%20optimize%20curve_fit#scipy.optimize.curve_fit) 的图形界面。当前，仅支持 Levenberg-Marquardt 优化器。此 GUI 基于 PyQt5。

## 安装

你可以从 [PyPi](https://pypi.org/project/curvefitgui/) 安装 `curvefitgui`：

```bash
pip install curvefitgui
```

此 GUI 支持 Python 3.7 及以上版本。  
**注意**：仅安装 `curvefitgui`，不包含任何必需的依赖项。根据你使用的是 pip 还是 conda 来管理环境，你需要手动安装以下额外的软件包：

- 使用 `pip`：

    ```bash
    pip install numpy scipy matplotlib PyQt5
    ```

- 使用 `conda`：

    ```bash
    conda install numpy scipy matplotlib qtpy pyqt
    ```

## 基本用法

使用 `curvefitgui.curve_fit_gui` 的一个简单示例：

```python
from curvefitgui import curve_fit_gui
import numpy as np

# 定义拟合函数
def f(x, a, b):
    '''
    线性拟合
    方程: y = ax + b
    a: 斜率
    b: 截距
    '''
    return a * x + b

# 定义 x 和 y 数据，作为长度相等的一维 numpy 数组
xdata = np.array([1, 2, 3, 4, 5])
ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
# 执行函数
curve_fit_gui(f, xdata, ydata)
```

## 参数

```python
popt, pcov = curve_fit_gui(f, xdata, ydata, xerr=None, yerr=None, p0=None, 
                           xlabel='x轴', ylabel='y轴', absolute_sigma=False,
                           jac=None, showgui=True, **kwargs)
```

`curve_fit_gui` 接受以下参数：
- **`f`：** 可调用
  拟合函数的定义。`f` 的第一个参数应为自变量；其他参数（至少一个）被认为是拟合参数。
- **`xdata`：** 一维 numpy 数组
  数据的 x 坐标
- **`ydata`：** 一维 numpy 数组
  数据的 y 坐标

`curve_fit_gui` 接受以下关键词参数：

- **`yerr`：** 一维 numpy 数组，可选（默认值：None）
  y 值的不确定性/误差，用于加权拟合，权重相对值为 1/yerr**2  
  （为了兼容，也可以使用关键词 `sigma` 来指定相同的内容）               
- **`xerr`：** 一维 numpy 数组，可选（默认值：None）
  x 值的误差，仅用于绘制误差条，拟合时被忽略                      
- **`xlabel`：** 字符串，可选（默认值：'x 值'）
  图中的 x 轴标题
- **`ylabel`：** 字符串，可选（默认值：'y 值'）
  图中的 y 轴标题
- **`p0`：** 类数组，可选
  拟合参数的初始值，如果未指定，则每个参数使用 1
- **`showgui`：** 布尔值，可选（默认值：True）
  如果为 True，则显示 GUI，否则不显示
- **`absolute_sigma`：** 布尔值，可选
  参见 scipy.optimize.curve_fit() 文档
- **`jac`：** 可调用，可选
  参见 scipy.optimize.curve_fit() 文档
- **`kwargs`：**
  兼容性关键词参数（例如，可以使用 `sigma` 指定 y 值的误差）

## 返回值

- **`popt`：** 成功拟合时，最小化平方残差的拟合参数值，否则返回 *None*。
- **`pcov`：** 拟合参数 `popt` 的协方差估计。  
（参见：[scipy.optimise.curve_fit API 参考](https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.curve_fit.html?highlight=scipy%20optimize%20curve_fit#scipy.optimize.curve_fit)）

## GUI 界面

一旦执行 `gui`，将会显示如下窗口。图下方描述了各个控件的功能。

![图形界面](https://github.com/moosepy/curvefitgui/raw/master/images/curvefitgui2.png)

### GUI 控件

1. **数据图：** matplotlib 绘制的数据图，显示为实心点，同时显示 y 误差和 x 误差条（如果提供）。如果进行了拟合，拟合曲线将显示为虚线。
2. **残差图：** matplotlib 绘制的残差图，显示测量值与拟合值的差异：`residual = ydata - f(xdata, *fitparameters)`
3. **模型设置：** 你可以在此处输入拟合参数的初始值。通过勾选 `fix` 复选框，可以将某个参数设置为固定值，例如，该参数在拟合过程中不被优化。
4. **权重设置：** 如果通过关键词参数 `yerr` 传递了 y 值的误差数据，你可以使用下拉框设置如何处理误差数据：
    - *None*：忽略误差数据
    - *Relative*：使用误差数据作为相对权重。对应于设置 scipy 的 curve_fit() 函数关键词 `absolute_sigma = False`。
    - *标准差*：将误差数据视为标准差。对应于设置 scipy 的 curve_fit() 函数关键词 `absolute_sigma = True`。
5. **评估：** 使用此按钮，根据当前设置的参数值计算模型函数。
6. **拟合：** 执行拟合并更新参数值。
7. **报告：** 进行拟合后，结果显示在此处。关于模型的信息实际上是传递给 `curvefitgui` 函数的 `f` 函数的 docstring。
8. **退出：** 退出 GUI 并返回拟合参数 `popt` 和 `pcov`。
9. **工具栏：** 这是标准的 matplotlib 工具栏，用于调整一些图形属性，提供缩放/平移和保存选项。
10. **拟合文本框：** 当进行有效的拟合时，生成此文本框。可以用鼠标将其移动到图中的任意方便位置。
11. **范围选择器：** 激活/停用范围选择器。范围选择器允许选择用于拟合的数据范围。仅考虑在两个垂直虚线之间的数据点进行拟合。可以使用鼠标移动虚线。