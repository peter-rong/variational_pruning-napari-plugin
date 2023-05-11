# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:32:30 2022

@author: Yigan
"""
import napari
# import sys
from qtpy.QtWidgets import QWidget, QCheckBox, QPushButton, QSlider, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
from .display import Display
from . import mainalgo

main_widget = "main"
debug_widget = "debug"


class WidgetManager:
    __instance = None

    def inst():
        if WidgetManager.__instance is None:
            WidgetManager.__instance = WidgetManager()
        return WidgetManager.__instance

    def __init__(self):
        self.widgets = list()

    def start(self):
        '''
        for w in self.widgets:
            w.sync()
        '''
        print("started")

    def add(self, widget: QWidget):
        self.widgets.append(widget)

    def find(self, name: str) -> QWidget:
        for w in self.widgets:
            if w.name == name:
                return w
        return None


class MainWidget(QWidget):

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.name = main_widget

        self.runButton = QPushButton(self)
        self.runButton.setText("Load Image")
        self.runButton.clicked.connect(MainWidget.run)
        self.runButton.move(0, 10)

        s, t = self.__make_slider_label()
        self.biSlider = s
        self.biSText = t

        self.biSlider.setRange(0, 100)

        self.biSlider.valueChanged.connect(self.set_bi_thr)
        self.biSlider.sliderReleased.connect(self.set_bithr_lift)
        self.biSlider.move(0, 40)
        self.biSText.move(100, 40)
        self.biSText.setText("Binary: " + str(self.biSlider.value()) +"%")

        #va

        self.valabel = QPushButton(self)
        self.valabel.setText("VA")
        self.valabel.move(10, 80)

        self.va_color_button = QPushButton(self)
        self.va_color_button.setText("Color")
        self.va_color_button.clicked.connect(MainWidget.va_color)

        self.va_graph_button = QPushButton(self)
        self.va_graph_button.setText("Prune")
        self.va_graph_button.clicked.connect(MainWidget.va_prune)

        self.va_color_button.move(60,80)
        self.va_graph_button.move(120,80)

        s, t = self.__make_slider_label()
        self.vaSlider = s
        self.vaSText = t

        self.vaSlider.setRange(0,110)

        self.vaSlider.valueChanged.connect(self.set_va_thr)
        self.vaSlider.sliderReleased.connect(self.set_vathr_lift)
        self.vaSlider.move(0, 140)
        self.vaSText.move(100, 140)
        self.vaSText.setText("Alpha: " + str(self.vaSlider.value()) + "degree")

        #et
        self.etlabel = QPushButton(self)
        self.etlabel.setText("ET")
        self.etlabel.move(10, 200)

        self.et_color_button = QPushButton(self)
        self.et_color_button.setText("Color")
        self.et_color_button.clicked.connect(MainWidget.et_color)

        self.et_graph_button = QPushButton(self)
        self.et_graph_button.setText("Prune")
        self.et_graph_button.clicked.connect(MainWidget.et_prune)

        self.et_color_button.move(60, 200)
        self.et_graph_button.move(120, 200)

        s, t = self.__make_slider_label()
        self.etSlider = s
        self.etSText = t

        self.etSlider.setRange(0, 100)

        self.etSlider.valueChanged.connect(self.set_et_thr)
        self.etSlider.sliderReleased.connect(self.set_etthr_lift)
        self.etSlider.move(0, 260)
        self.etSText.move(100, 260)
        self.etSText.setText("Threshold: " + str(self.etSlider.value()) + "%")

        WidgetManager.inst().add(self)

    def run():
        WidgetManager.inst().start()
        mainalgo.SkeletonApp.inst().run()

    def va_color():
        mainalgo.SkeletonApp.inst().va_color()

    def va_prune():
        mainalgo.SkeletonApp.inst().va_prune()

    def set_bi_thr(self):
        self.biSText.setText("Binary: " + str(self.biSlider.value()) +"%")

    def set_bithr_lift(self):
        mainalgo.SkeletonApp.inst().reset_bithresh(self.biSlider.value())

    def set_va_thr(self):
        self.vaSText.setText("Alpha: " + str(self.vaSlider.value()) + " degree")

    def set_vathr_lift(self):
        mainalgo.SkeletonApp.inst().reset_vathresh(self.vaSlider.value())

    def et_color():
        mainalgo.SkeletonApp.inst().et_color()

    def et_prune():
        mainalgo.SkeletonApp.inst().et_prune()

    def set_et_thr(self):
        self.etSText.setText("Threshold: " + str(self.etSlider.value()) + " %")

    def set_etthr_lift(self):
        mainalgo.SkeletonApp.inst().reset_etthresh(self.etSlider.value())

    def __make_slider_label(self):
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 100)
        sText = QLabel('0', self)
        sText.setMinimumWidth(80)
        return slider, sText

class DebugWidget(QWidget):
    """Any QtWidgets.QWidget or magicgui.widgets.Widget subclass can be used."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)

        self.name = debug_widget

        self.show_edge_box = self.__make_box("show boundary", 0)
        self.show_vor_box = self.__make_box("show full voronoi", 40)
        self.show_intvor_box = self.__make_box("show internal voronoi", 80)
        self.show_hm_box = self.__make_box("show heatmap", 120)
        self.show_bt_box = self.__make_box("show burn time", 160)
        self.show_et_box = self.__make_box("show et", 200)
        self.show_final_box = self.__make_box("show final", 240)
        self.show_angle_box = self.__make_box("show angle", 280)
        self.show_thickness_box = self.__make_box("show thickness", 320)
        self.show_PCST_box = self.__make_box("show PCST", 360)
        self.show_PCST_result_box = self.__make_box("show PCST result", 400)
        self.show_skeleton_result_box = self.__make_box("show skeleton result", 440)
        self.show_dynamic_box = self.__make_box("show dynamic", 480)
        self.show_full_dynamic_box = self.__make_box("show full dynamic", 520)
        self.output_skeleton = self.__make_box("output skeleton graph", 560)

        WidgetManager.inst().add(self)

    def sync(self):
        config = Display.current().config
        config.show_edgepoints = self.show_edge_box.isChecked()
        config.show_voronoi = self.show_vor_box.isChecked()
        config.show_internal_voronoi = self.show_intvor_box.isChecked()
        config.show_heatmap = self.show_hm_box.isChecked()
        config.show_bt = self.show_bt_box.isChecked()
        config.show_et = self.show_et_box.isChecked()
        config.show_final = self.show_final_box.isChecked()
        config.show_angle = self.show_angle_box.isChecked()
        config.show_thickness = self.show_thickness_box.isChecked()
        config.show_pcst = self.show_PCST_box.isChecked()
        config.show_pcst_result = self.show_PCST_result_box.isChecked()
        config.show_skeleton_result = self.show_skeleton_result_box.isChecked()
        config.show_dynamic = self.show_dynamic_box.isChecked()
        config.show_full_dynamic = self.show_full_dynamic_box.isChecked()
        config.output_skeleton = self.output_skeleton.isChecked()

        Display.current().set_config(config)

    def __make_box(self, text, position):
        box = QCheckBox(self)
        box.setText(text)
        box.move(0, position)
        return box


'''
app = QApplication(sys.argv)
w = QWidget()
w.resize(300,300)
w.setWindowTitle("HA")

label.setText("Behold the Guru, Guru99")
label = QLabel(w)

label.move(100,130)
label.show()

w.show()

sys.exit(app.exec_())
'''
