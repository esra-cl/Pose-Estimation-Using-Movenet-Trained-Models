from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt,pyqtSignal
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)















    
    def make_pred(self,img, keypoints_dict, label):
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img[0])
        plt.subplot(1, 3, 2)

        plt.imshow(img[0])
        plt.title('Pose')
        plt.axis('off')
        for i in range(13):
            plt.scatter(keypoints_dict[label[i]][1],keypoints_dict[self.label[i]][0],color='green')

        connections = [
            ('nose', 'left eye'), ('left eye', 'left ear'), ('nose', 'right eye'), ('right eye', 'right ear'),
            ('nose', 'left shoulder'), ('left shoulder', 'left elbow'), ('left elbow', 'left wrist'),
            ('nose', 'right shoulder'), ('right shoulder', 'right elbow'), ('right elbow', 'right wrist'),
            ('left shoulder', 'left hip'), ('right shoulder', 'right hip'), ('left hip', 'right hip'),
            ('left hip', 'left knee'), ('right hip', 'right knee'),('left knee','left ankle'),('right knee','right ankle')
        ]
        for start_key, end_key in connections:
            if start_key in keypoints_dict and end_key in keypoints_dict:
                start_point = keypoints_dict[start_key][:2]  # Take first two values
                end_point = keypoints_dict[end_key][:2]      # Take first two values
                plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)
        
        plt.subplot(1, 3, 3)
        plt.imshow((img[0]/255)/255)
        plt.title('Only Pose Image')
        for start_key, end_key in connections:
            if start_key in keypoints_dict and end_key in keypoints_dict:
                start_point = keypoints_dict[start_key][:2]  # Take first two values
                end_point = keypoints_dict[end_key][:2]      # Take first two values
                plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)
