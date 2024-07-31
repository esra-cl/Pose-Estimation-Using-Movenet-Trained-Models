from application import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QScrollArea, QWidget, QVBoxLayout, QLabel, QPushButton
import sys
import cv2 as cv
import os 
from PyQt5.QtCore import QThread, pyqtSignal, QObject,Qt
from pose_estimation import PoseEstimation
from clickable_label import ClickableLabel
#python -m PyQt5.uic.pyuic -o application.py application.ui

class App(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()  # Corrected super call
        self.setupUi(self)
        self.video_path_to_play=""
        self.videos_paths = []
        self.to_be_opened = None
        # set threads 
        self.worker = WorkerThread(self, 0)
        self.worker01= WorkerThread(self,1)
        self.worker02= WorkerThread(self,2)
        self.display()

    def display(self):
        # Set scroll area
        self.scroll_area = QScrollArea(self.groupBox)
        self.scroll_area.setGeometry(self.opened.pos().x(), 30,290,379)
        self.container_widget = QWidget()
        self.layout = QVBoxLayout(self.container_widget)
        self.scroll_area.setWidget(self.container_widget)
        self.scroll_area.setWidgetResizable(True)
        # Connect buttons to slots 
        self.choose_file_2.clicked.connect(self.choose_files)
        self.play_2.clicked.connect(self.ply_video)
        self.rgb_pose.clicked.connect(self.rgb_pose_checkBox_clicked)
        self.pose.clicked.connect(self.pose_checkBox_clicked)
        self.pose_extracte_2.clicked.connect(self.extracte_pose)

    def ply_video(self):
        self.worker01.start()
    def extracte_pose(self):
        self.worker02.start()

    def rgb_pose_checkBox_clicked (self):
        self.pose.setCheckState(False)
    def pose_checkBox_clicked (self):
        self.rgb_pose.setCheckState(False)
    def choose_files(self):
        self.clear_()
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Videos", "", "Video Files (*.mp4 *.avi *.mkv *.jpg *.png)", options=options)
        if files:
            for file_name in files:
                self.videos_paths.append(file_name)
                self.create_labels_for_video_names(file_name)
        self.play_2.setEnabled(True)
    def clear_(self):
        self.videos_paths = []
        self.play_2.setEnabled(False)


    def create_labels_for_video_names (self,video_path):
        base_name = os.path.basename(video_path)
        i = len(self.container_widget.findChildren(QLabel))
        new_label = ClickableLabel(f"{i + 1}. {base_name}", self.container_widget)
        new_label.clicked.connect(self.label_clicked)
        self.layout.addWidget(new_label)

    def label_clicked(self):
        label = self.sender()
        dirname_=os.path.dirname( self.videos_paths[-1])
        strin_ = str(label.text())
        pattern = r"\d+\.\s(.+\.\w+)"
        import re 
        match = re.search(pattern, strin_)
        file_name = match.group(1)
        self.video_path_to_play = fr"{dirname_}/{file_name}"
        print(self.video_path_to_play)

    def extract_from_worker_thread(self):
        pass 
class WorkerThread(QThread):
    
    def __init__(self, main_window, flag):
        super().__init__()
        self.main_window = main_window
        self.flag = flag
        self.isToExtract= None
        self.pose=None
        self.rgp_pose=None
        self.ps = PoseEstimation()
    def run(self):
        if self.flag == 0:
            pass
        elif self.flag == 1 :
            self.rgp_pose = self.main_window.rgb_pose.isChecked()
            self.pose = self.main_window.pose.isChecked()
            print(self.main_window.video_path_to_play)
            self.pose_est(self.main_window.video_path_to_play,False)
        elif self.flag== 2 : 
                self.rgp_pose = False
                self.pose = True
                for file_path in self.main_window.videos_paths :
                    self.pose_est(file_path,True)

    def pose_est(self,file_path,isToExtracte):
        self.ps.pose_estimation_predict(self.main_window.captured ,self.rgp_pose,self.pose,file_path,isToExtracte)

if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(application.exec_())
