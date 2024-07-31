import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import threading


class Thread_(threading.Thread):
    def save_to_file(self, keypoint_frames, video_path, i):
        file_name = os.path.basename(video_path).replace("mp4", "txt")
        path_ = r"C:\Users\HP\Downloads\feature_extractor\Pose-Estimation-Using-Movenet-Trained-Models\pose_estimation\data"
        
        txt_toBe_created = os.path.join(path_, file_name)
        os.makedirs(os.path.dirname(txt_toBe_created), exist_ok=True)
        with open(txt_toBe_created,'a') as f:
            f.write(f"{i}")
            for idx, (_,item) in enumerate(keypoint_frames.items()):
                x, y = item[:2]
                f.write(f"{x} {y} ")
            f.write("\n")


class PoseEstimation:
    def __init__(self):
        self.interpreter = None
        self.keypoints_with_scores = None
        self.input_details = None
        self.output_details = None
        self.keypoints_dict = {}


        self.thread = Thread_()
        # Start the thread
        self.thread.start()
        self.model_path = r'C:\Users\HP\Downloads\feature_extractor\Pose-Estimation-Using-Movenet-Trained-Models\pose_estimation\movenet-tflite-singlepose-thunder-tflite-float16-v1\4.tflite'
        self.labels = ["nose", "left eye", "right eye", "left ear", "right ear",
                       "left shoulder", "right shoulder", "left elbow", "right elbow",
                       "left wrist", "right wrist", "left hip", "right hip",
                       "left knee", "right knee", "left ankle", "right ankle"]

        self.connections = {"nose_leftEye": (0, 1),
                            "nose_rightEye": [0, 2],
                            "leftEye_leftEar": [1, 3],
                            "rightEye_rightEar": [2, 4],
                            "nose_leftShoulder": [0, 5],
                            "nose_rightShoulder": [0, 6],
                            "leftShoulder_leftElbow": [5, 7],
                            "leftElbow_leftWrist": [7, 9],
                            "rightShoulder_rightElbow": [6, 8],
                            "rightElbow_rightWrist": [8, 10],
                            "leftShoulder_leftHip": [5, 11],
                            "rightShoulder_rightHip": [6, 12],
                            "leftHip_righthip": [11, 12],
                            "leftHip_leftKnee": [11, 13],
                            "lefKnee_leftAnkle": [13, 15],
                            "rightHip_rightKnee": [12, 14],
                            "rightKnee_rightAnkle": [14, 16],
                            }
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"The model file does not exist at the specified path: {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def image_preparation(self, image):
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_pad(image, 256, 256)
        image = tf.cast(image, dtype=tf.uint8)
        return image
    
    def pose_estimation_predict(self,label,rgb_flag,pose_flag,file_path,isToExtracte):
        cap = cv.VideoCapture(file_path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            input_image = self.image_preparation(frame)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
            self.interpreter.invoke()
            self.keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.keypoints_dict = {self.labels[idx]: keypoint[:] for idx, keypoint in enumerate(self.keypoints_with_scores[0][0])}
            self.Draw_on_frame(frame, self.keypoints_with_scores, isToExtracte,file_path,i)
            pose_only_frame = self.Draw_pose_only(frame, self.keypoints_with_scores)
            rgb_skeleton_frame = cv.resize(frame, (361, 271))
            pose_only_frame = cv.resize(pose_only_frame, (361, 271))
            from PyQt5.QtGui import QImage ,QPixmap
            if rgb_flag:
                rgb_skeleton_frame = cv.cvtColor(rgb_skeleton_frame, cv.COLOR_BGR2RGB)
                h, w, ch = rgb_skeleton_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_skeleton_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)            
                pixmap = QPixmap.fromImage(q_image)
                label.setPixmap(pixmap)
            elif pose_flag:
                pose_only_frame = cv.cvtColor(pose_only_frame, cv.COLOR_BGR2RGB)
                h, w, ch = pose_only_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(pose_only_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)            
                pixmap = QPixmap.fromImage(q_image)
                label.setPixmap(pixmap)

    def Draw_on_frame(self, frame, keypoints,isToExtracte,file_path, i, ts_=0.4):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for _, connection in self.connections.items():
            joint1, joint2 = connection
            y1, x1, ac1 = shaped[joint1]
            y2, x2, ac2 = shaped[joint2]
            if ac1 >= ts_ and ac2 >= ts_:
                cv.circle(frame, (int(x1), int(y1)), 10, (223, 0, 223), -1)
                cv.circle(frame, (int(x2), int(y2)), 10, (223, 0, 223), -1)
                
                cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 128), 5)
                if isToExtracte:
                    self.thread.save_to_file(self.keypoints_dict,file_path, i)
                    i += 1
    def Draw_pose_only(self, frame, keypoints, ts_=0.4):
        y, x, _ = frame.shape
        pose_only_frame = np.zeros((y, x, 3), dtype=np.uint8)
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        for _, connection in self.connections.items():
            joint1, joint2 = connection
            y1, x1, ac1 = shaped[joint1]
            y2, x2, ac2 = shaped[joint2]
            if ac1 >= 0.4 and ac2 >= 0.4:
                cv.circle(pose_only_frame, (int(x1), int(y1)), 10, (223, 0, 223), -1)
                cv.circle(pose_only_frame, (int(x2), int(y2)), 10, (223, 0, 223), -1)
                cv.line(pose_only_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 128), 5)
        return pose_only_frame

