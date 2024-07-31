import tensorflow as tf
import numpy as np
import cv2
import os
from PySide6.QtCore import QObject
from PySide6.QtWidgets import *

class Movenet(QObject):
    def __init__(self, v_path):
        super().__init__()
        self.path = 'C:\\Users\\User\\Desktop\\ortakproje\\'
        print(v_path)
        converted_path=str(v_path)[len(str(self.path)):]
        self.video_path = converted_path
        print(self.video_path)
        self.interpreter = tf.lite.Interpreter("lite-model_movenet_singlepose_lightning_3.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.EDGES = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }
        self.keypoints_list = []
        self.keypoints_with_scores=None
        self.selektion()

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) and (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    def selektion(self):
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()

            if not ret:
                break 
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], np.array(input_image))
            self.interpreter.invoke()
            self.keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

            self.draw_connections(frame, self.keypoints_with_scores, self.EDGES, 0.4)
            self.draw_keypoints(frame, self.keypoints_with_scores, 0.4)

            small_frame = cv2.resize(frame, (800, 600))  # Adjust the size as needed

            # Display the resized frame with keypoints
            cv2.imshow("Movenet Keypoints", small_frame)
            
            # listeye alindi
            for i, keypoint in enumerate(np.squeeze(self.keypoints_with_scores)):
                x, y, score = keypoint[:3]
                self.keypoints_list.append(f"{i}, {x}, {y}, {score}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.save_keypoints_to_file()
        
        cap.release()
        cv2.destroyAllWindows()


    def save_keypoints_to_file(self):
        path=self.video_path.replace("mp4","txt")
        path=path.replace("raw","data")

        os.makedirs(os.path.dirname(self.path + path), exist_ok=True)

        with open(self.path+path, 'w') as f:
            f.write(", ".join(self.keypoints_list))

        print("Değerler kaydedildi")
        self.show_movenet_message()

    def show_movenet_message(self):
        mesajobje = QMessageBox()
        mesajobje.setIcon(QMessageBox.Information)
        mesajobje.setText("Texte Yazdırıldı.          ")
        mesajobje.setWindowTitle("Movenet")
        mesajobje.setStandardButtons(QMessageBox.Ok)
        mesajobje.setEscapeButton(QMessageBox.Ok)
        mesajobje.button(QMessageBox.Ok).setText("Tamam")
        mesajobje.exec_()





