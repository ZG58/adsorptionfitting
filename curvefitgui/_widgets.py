# -*- coding: utf-8 -*-

# 导入所需的包
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# Matplotlib 包
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import rcParams

from ._settings import settings
from ._tools import float_to_str

from _model import langmuir,toth

rcParams['mathtext.fontset'] = 'cm'


class DraggableVLine:
    """ 创建一个可拖动的垂直线类 """

    lock = None  # 我们需要这个来确保一次只能拖动一条线

    def __init__(self, ax, x, linewidth=4, linestyle='--', color='gray'):
        self.line = ax.axvline(x=x, linewidth=linewidth, linestyle=linestyle, color=color)
        self.press = None
        self.connect()

    def get_pos(self):
        return self.line.get_xdata()[0]

    def remove(self):
        self.line.remove()

    def connect(self):
        '连接所有我们需要的事件'
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        if DraggableVLine.lock is not None: return
        contains, _ = self.line.contains(event)
        if not contains: return
        x, _ = self.line.get_xdata()
        self.press = x, event.xdata
        DraggableVLine.lock = self

    def on_motion(self, event):
        if self.press is None: return
        if DraggableVLine.lock is not self: return
        if event.inaxes != self.line.axes: return
        x, xpress = self.press
        dx = event.xdata - xpress
        x_clip = x + dx
        self.line.set_xdata([x_clip, x_clip])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        if DraggableVLine.lock is not self: return
        DraggableVLine.lock = None
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


class RangeSelector:
    """ 创建一个包含两条可拖动垂直线的范围选择器类 """

    def __init__(self, ax, pos1, pos2):
        self.ax = ax  # 持有线条的坐标轴
        self.pos = [pos1, pos2]  # 线条的初始位置
        self.drag_lines = [DraggableVLine(self.ax, pos) for pos in self.pos]

    def get_range(self):
        pos = [dragline.get_pos() for dragline in self.drag_lines]
        pos.sort()
        return pos

    def remove(self):
        for dragline in self.drag_lines:
            dragline.remove()


class PlotWidget(QtWidgets.QWidget):
    """ Qt 小部件，用于容纳 matplotlib 画布和与图形交互的工具 """

    resized = QtCore.pyqtSignal()  # 当小部件被调整大小时发射信号

    def __init__(self, data, xlabel, ylabel):
        QtWidgets.QWidget.__init__(self)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = PlotCanvas(data, xlabel, ylabel)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.addSeparator()

        self.ACshowselector = QtWidgets.QAction('激活/清除范围选择')
        self.ACshowselector.setIconText('范围选择')
        self.ACshowselector.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.ACshowselector.triggered.connect(self._toggle_showselector)

        self.toolbar.addAction(self.ACshowselector)

        self.toolbar.addSeparator()
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

        self.resized.connect(self.update_plot)  # 当窗口调整大小时更新图形，以适应窗口

    def resizeEvent(self, event):
        self.resized.emit()  # 发射 resized 信号
        return super(PlotWidget, self).resizeEvent(event)

    def update_plot(self):
        self.canvas.update_plot()  # 更新图形

    def _toggle_showselector(self):
        self.canvas.toggle_rangeselector()  # 切换范围选择器的显示状态


class PlotCanvas(FigureCanvas):
    """ 类用于持有带有 matplotlib 图形和两个子图的画布，用于绘制数据和残差 """

    def __init__(self, data, xlabel, ylabel):
        self.data = data  # 包含 x、y 和误差数据
        self.fitline = None  # 如果可用，则包含拟合线
        self.residuals = None  # 如果可用，则包含残差

        # 设置 FigureCanvas
        self.fig = Figure(dpi=settings['FIG_DPI'], tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # 初始化一些状态变量
        self.range_selector = None

        # 创建图形和坐标轴
        gs = self.fig.add_gridspec(3, 1)  # 定义三行一列

        # 需要先创建 ax2 以防与 ax1 相关的文本框出现在残差图后面
        self.ax2 = self.fig.add_subplot(gs[2, 0])  # ax2 持有残差图，并占据一行
        self.ax1 = self.fig.add_subplot(gs[0:2, 0], sharex=self.ax2)  # ax1 持有数据图，并占据两行

        self.ax1.grid()
        self.ax2.grid()
        self.ax1.set_ylabel(ylabel, fontname=settings['TEXT_FONT'], fontsize=settings['TEXT_SIZE'])
        self.ax2.set_ylabel('Residual', fontname=settings['TEXT_FONT'], fontsize=settings['TEXT_SIZE'])
        self.ax2.set_xlabel(xlabel, fontname=settings['TEXT_FONT'], fontsize=settings['TEXT_SIZE'])

        # 创建空的线条用于数据、拟合和残差
        self.data_line, = self.ax1.plot([], [], color='black', marker='o', fillstyle='none', lw=0, label='data')
        self.fitted_line, = self.ax1.plot([], [], label='fitting curve', linestyle='--', color='black')
        self.residual_line, = self.ax2.plot([], [], color='k', marker='.', lw=1)
        self.zero_res = None  # 用于指示残差图中零的虚线的持有者

        # 创建图例
        self.ax1.legend(loc='best', fancybox=True, framealpha=0.5,
                        prop={'family': settings['TEXT_FONT'], 'size': settings['TEXT_SIZE']})

        # 创建一个注释框来保存拟合结果
        bbox_args = dict(boxstyle=patches.BoxStyle("round", pad=0.5), fc="0.9", alpha=0.5)
        self.result_box = self.ax1.annotate('', xy=(0.5, 0.5), xycoords='axes fraction', fontname=settings['TEXT_FONT'],
                                            size=settings['TEXT_SIZE'], bbox=bbox_args)
        self.result_box.draggable()

        # 填充绘图线
        self.data_line.set_data(self.data.x, self.data.y)

        # 如果需要，创建误差条
        if self.data.ye is not None:
            self.yerrobar = self.ax1.errorbar(self.data.x, self.data.y, yerr=self.data.ye,
                                              fmt='none', color=settings['BAR_Y_COLOR'],
                                              elinewidth=settings['BAR_Y_THICKNESS'],
                                              capsize=2)
        if self.data.xe is not None:
            self.xerrobar = self.ax1.errorbar(self.data.x, self.data.y, xerr=self.data.xe,
                                              fmt='none', color=settings['BAR_X_COLOR'],
                                              elinewidth=settings['BAR_X_THICKNESS'],
                                              capsize=2)

            # 设置刻度标签属性
        for labels in [self.ax1.get_xticklabels(), self.ax1.get_yticklabels(),
                       self.ax2.get_xticklabels(), self.ax2.get_yticklabels()]:
            for tick in labels:
                tick.set_color(settings['TICK_COLOR'])
                tick.set_fontproperties(settings['TICK_FONT'])
                tick.set_fontsize(settings['TICK_SIZE'])

    def set_results_box(self, text, loc):
        self.result_box.set_text(text)
        self.result_box.set_visible(True)

    def disable_results_box(self):
        self.result_box.set_visible(False)

    def toggle_rangeselector(self):
        if self.range_selector is None:
            self.range_selector = RangeSelector(self.ax1, np.min(self.data.x), np.max(self.data.x))
            self.redraw()
        else:
            self.range_selector.remove()
            self.range_selector = None
            self.redraw()

    def set_residuals(self, residuals):
        self.residuals = residuals

    def set_fitline(self, fitline):
        self.fitline = fitline

    def get_range(self):
        if self.range_selector is None:
            self.data.set_mask(-np.inf, np.inf)
        else:
            self.data.set_mask(*self.range_selector.get_range())

    def update_plot(self):
        # 更新残差和/或拟合线（如果存在）

        if self.residuals is not None:
            # 如果零残差线尚未创建，则创建它
            if self.zero_res is None:
                self.ax2.axhline(y=0, linestyle='--', color='black')

            # 如果需要，排序数据
            if settings['SORT_RESIDUALS']:
                order = np.argsort(self.data.x)
            else:
                order = np.arange(0, len(self.data.x))

            self.residual_line.set_data(self.data.x[order], self.residuals[order])

        if self.fitline is not None:
            self.fitted_line.set_data(self.fitline[0], self.fitline[1])

        # 重新调整坐标轴
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax2.relim()
        self.ax2.autoscale()

        # 使残差图的最小和最大 y 轴限制相等
        ymax = max(np.abs(self.ax2.get_ylim()))
        self.ax2.set_ylim(-ymax, ymax)

        # 绘制图形
        self.redraw()

    def redraw(self):
        # self.fig.canvas.draw()
        self.draw()


class ParamWidget(QtWidgets.QWidget):
    """ Qt 小部件，用于显示和更改拟合参数 """

    def __init__(self, par):
        QtWidgets.QWidget.__init__(self)
        self.par = par
        self.label = QtWidgets.QLabel(par.name)
        self.edit = QtWidgets.QLineEdit('')
        self.update_value()
        self.check = QtWidgets.QCheckBox('fix')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addWidget(self.check)
        self.setLayout(layout)

    def read_value(self):
        """ 读取用户输入（值和固定）到参数数据中 """
        self.par.value = float(self.edit.text())
        self.par.fixed = self.check.isChecked()
        return None

    def update_value(self):
        value = self.par.value
        self.edit.setText(float_to_str(value, settings['SIGNIFICANT_DIGITS']))
        return None


class ReportWidget(QtWidgets.QTextEdit):
    """ 在不可编辑的文本框中打印拟合报告。报告应该是一个（嵌套的）字典 """

    def __init__(self):
        QtWidgets.QTextEdit.__init__(self, 'none', )
        self.setFont(QtGui.QFont(settings['REPORT_FONT'], settings['REPORT_SIZE']))  # 设置字体和大小
        self.setReadOnly(True)  # 设置为只读

    def update_report(self, fitreport):
        """ 用（嵌套的）字典 fitreport 的内容更新文本框的文本 """

        def print_dict(adict, level):
            for key, item in adict.items():
                if type(item) is dict:
                    if level == 1: self.insertPlainText('========== ')  # 打印分隔线
                    self.insertPlainText(str(key))  # 打印键
                    if level == 1: self.insertPlainText(' ========== ')  # 打印分隔线
                    self.insertPlainText('\n\n')  # 换行
                    print_dict(item, level + 1)  # 递归打印子字典
                else:
                    if type(item) == np.float64:
                        item_str = float_to_str(item, settings['SIGNIFICANT_DIGITS'])  # 格式化浮点数
                    else:
                        item_str = str(item)
                    self.insertPlainText(str(key) + '\t\t: ' + item_str + '\n')  # 打印键值对
            self.insertPlainText('\n')  # 换行

        self.clear()  # 清空文本框
        print_dict(fitreport, 1)  # 打印报告


class ModelWidget(QtWidgets.QGroupBox):
    """ Qt 小部件，用于显示和控制拟合模型 """

    def __init__(self, model, weightoptions, model_selection):
        self.model = model  # 存储模型
        QtWidgets.QGroupBox.__init__(self, '模型配置')  # 初始化组框
        self.initGUI(weightoptions, model_selection)  # 初始化图形用户界面
        self.set_weight()  # 设置权重
        self.modelname:str = ''

    def initGUI(self, weightoptions, model_selection):
        VBox = QtWidgets.QVBoxLayout()  # 垂直布局
        HBox = QtWidgets.QHBoxLayout()  # 水平布局
        self.parviews = [ParamWidget(par) for par in self.model.fitpars]  # 创建参数小部件
        self.WeightLabel = QtWidgets.QLabel('Weighted:')  # 权重拟合标签
        self.ModelLabel = QtWidgets.QLabel('Models:')  # 权重拟合标签
        self.Yweightcombobox = QtWidgets.QComboBox()  # 权重下拉框
        self.model_selector = QtWidgets.QComboBox()  # 模型下拉框
        self.Yweightcombobox.addItems(weightoptions)  # 添加权重选项
        self.model_selector.addItems(model_selection)  # 添加模型选项

        HBox.addWidget(self.WeightLabel)  # 添加标签到水平布局
        HBox.addWidget(self.Yweightcombobox)  # 添加下拉框到水平布局
        HBox.addWidget(self.ModelLabel)  # 添加标签到水平布局
        HBox.addWidget(self.model_selector)  # 添加下拉框到水平布局
        HBox.addStretch(1)  # 添加伸缩项
        for parview in self.parviews:
            VBox.addWidget(parview)  # 将参数小部件添加到垂直布局
        VBox.addLayout(HBox)  # 添加水平布局到垂直布局
        self.setLayout(VBox)  # 设置布局
        return None

    def disable_weight(self):
        self.Yweightcombobox.setDisabled(True)  # 禁用权重下拉框

    def enable_weight(self):
        self.Yweightcombobox.setEnabled(True)  # 启用权重下拉框

    def get_weight(self):
        return self.Yweightcombobox.currentText()  # 获取当前选中的权重

    def get_model(self):
        return self.model_selector.currentText()  # 获取当前选中的模型

    def set_weight(self):
        index = self.Yweightcombobox.findText(self.model.weight, QtCore.Qt.MatchFixedString)  # 查找权重的索引
        if index >= 0:
            self.Yweightcombobox.setCurrentIndex(index)  # 设置当前选中的权重

    def read_values(self):
        """ 从用户输入读取值到模型中 """
        for parview in self.parviews:
            parview.read_value()  # 读取每个参数的值
        self.model.weight = self.get_weight()  # 获取权重并存入模型
        self.modelname = self.get_model()

        return None

    def update_values(self):
        for parview in self.parviews:
            parview.update_value()  # 更新每个参数的值
        return None
