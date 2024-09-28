# 导入所需的包
import warnings
import sys
from scipy.optimize import OptimizeWarning
from PyQt5 import QtCore, QtWidgets

from ._tools import Fitter, value_to_string
from ._widgets import PlotWidget, ModelWidget, ReportWidget
from ._settings import settings
from ._version import __version__ as CFGversion


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, afitter, xlabel, ylabel):
        super(MainWindow, self).__init__()

        # 执行一些初始默认设置
        self.fitter = afitter
        self.xlabel, self.ylabel = xlabel, ylabel
        self.output = (None, None)
        self.xerrorwarning = settings['XERRORWARNING']

        self.initGUI()

        self.plotwidget.update_plot()

    def closeEvent(self, event):
        """在IPython控制台/Spyder IDE中正常退出所需的方法"""
        QtWidgets.QApplication.quit()

    def initGUI(self):
        # 主GUI属性
        self.setGeometry(100, 100, 1415, 900)
        self.setWindowTitle('等温线拟合 ' + CFGversion)
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # 创建所需的小部件
        self.plotwidget = PlotWidget(self.fitter.data, self.xlabel, self.ylabel)  # 显示图形
        self.modelview = ModelWidget(self.fitter.model, self.fitter.get_weightoptions(), self.fitter.get_models())  # 显示模型并允许用户设置拟合属性
        self.fitbutton = QtWidgets.QPushButton('拟合', clicked=self.fit)
        self.evalbutton = QtWidgets.QPushButton('评估', clicked=self.evaluate)
        self.reportview = ReportWidget()  # 显示拟合结果
        self.quitbutton = QtWidgets.QPushButton('退出', clicked=self.close)

        # 创建按钮的布局
        self.buttons = QtWidgets.QGroupBox()
        buttonslayout = QtWidgets.QHBoxLayout()
        buttonslayout.addWidget(self.evalbutton)
        buttonslayout.addWidget(self.fitbutton)
        self.buttons.setLayout(buttonslayout)

        # 创建一个带有垂直布局的框架以组织模型视图、拟合按钮和报告视图
        self.fitcontrolframe = QtWidgets.QGroupBox()
        fitcontrollayout = QtWidgets.QVBoxLayout()
        for widget in (self.modelview, self.buttons, self.reportview, self.quitbutton):
            fitcontrollayout.addWidget(widget)
        self.fitcontrolframe.setLayout(fitcontrollayout)

        # 将所有组件整合在一起：设置主布局
        mainlayout = QtWidgets.QHBoxLayout(self._main)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.plotwidget)
        splitter.addWidget(self.fitcontrolframe)
        mainlayout.addWidget(splitter)

    def showdialog(self, message, icon, info='', details=''):
        """ 显示信息对话框 """
        msg = QtWidgets.QMessageBox()
        if icon == 'critical': msg.setIcon(QtWidgets.QMessageBox.Critical)
        if icon == 'warning': msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setInformativeText(info)
        msg.setWindowTitle("消息")
        msg.setDetailedText(details)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def set_output(self, output):
        """输出应为一个元组，包含在关闭应用时返回的变量"""
        self.output = output

    def get_output(self):
        """允许在应用关闭时返回当前存储的输出"""
        return self.output

    def evaluate(self):
        """ 更新模型并计算当前参数值下的模型曲线 """
        # 从用户输入更新模型值
        try:
            self.modelview.read_values()
        except ValueError:
            self.showdialog('无效的初始参数值输入', 'critical')
            return None

        # 评估
        self.reportview.update_report({})
        self.plotwidget.canvas.set_fitline(self.fitter.get_curve())
        self.plotwidget.canvas.set_residuals(self.fitter.get_residuals(check=False))
        self.plotwidget.canvas.disable_results_box()
        self.plotwidget.update_plot()

    def fit(self):
        """ 更新模型，执行拟合并用结果更新小部件 """
        # 从用户输入更新模型值
        try:
            self.modelview.read_values()
        except ValueError:
            self.showdialog('无效的初始参数值输入', 'critical')
            return None

        # 更新拟合范围
        self.plotwidget.canvas.get_range()

        # 显示x错误数据的警告
        if (self.fitter.data.xe is not None) and self.xerrorwarning:
            self.showdialog('x中的误差在拟合中被忽略！', 'warning')
            self.xerrorwarning = False

        # 执行拟合
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)  # 确保OptimizeWarning被抛出为异常
            try:
                fitpars, fitcov = self.fitter.fit()
            except (ValueError, RuntimeError, OptimizeWarning):
                self.showdialog(str(sys.exc_info()[1]), 'critical')

            else:
                # 更新输出
                self.set_output((fitpars, fitcov))

                # 更新小部件
                self.modelview.update_values()
                self.reportview.update_report(self.fitter.get_report())
                self.plotwidget.canvas.set_fitline(self.fitter.get_fitcurve())
                self.plotwidget.canvas.set_residuals(self.fitter.get_residuals())
                self.plotwidget.canvas.set_results_box(self._get_result_box_text(), 2)
                self.plotwidget.update_plot()

    def _get_result_box_text(self):
        text = 'Fitting result:'
        text = text + '\n' + 'weight:' + self.fitter.model.weight
        for par in self.fitter.model.fitpars:
            n = par.name
            v = par.value
            e = par.sigma
            f = par.fixed
            text = text + '\n' + value_to_string(n, v, e, f)
        return text


def execute_gui(f, xdata, ydata, xerr, yerr, p0, xlabel, ylabel,
                absolute_sigma, jac, showgui, **kwargs):
    """
    辅助函数，执行带有拟合器类实例的GUI
    """

    afitter = Fitter(f, xdata, ydata, xerr, yerr, p0, absolute_sigma, jac, **kwargs)
    if not showgui:
        return afitter.fit()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    MyApplication = MainWindow(afitter, xlabel, ylabel)
    MyApplication.show()
    app.exec_()
    return MyApplication.get_output()

if __name__ == "__main__":
    pass