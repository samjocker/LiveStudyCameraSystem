import cv2
import mediapipe as mp
from PIL import ImageGrab
import numpy as np
import serial
from time import sleep

COM_PORT = 'COM3'
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)
sleep(2)
print("序列阜連線成功")

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# cap = cv2.VideoCapture(0)
img_bgr = ImageGrab.grab()
width, height = img_bgr.size

cdn_x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cdn_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def tch_follower(x=[],y=[],max_x=0.72,min_x=0.28):
  if x[11]<=1 and x[12]<=1:
    middle_x = (x[11]+x[12])/2
    if middle_x:
      if middle_x <= min_x:
        return "left"
      elif middle_x >= max_x:
        return "right"
      else:
        return None

def write_space(x=[],y=[],R_max=-1,R_min=1.2,L_max=1,L_min=-1.2,hand_long=0.3):
  if x[11] and x[12] and x[15] and x[16] and y[11] and y[12] and y[15] and y[16] and y[23]:
    hand_m_right = (y[12]-y[16])/(x[12]-x[16])
    hand_m_left = (y[11]-y[15])/(x[11]-x[15])
    hand_d_right = ((y[12]-y[16])**2+(x[12]-x[16])**2)**0.5
    hand_d_left = ((y[11]-y[15])**2+(x[11]-x[15])**2)**0.5
    standard = y[23] - y[11]
    hand_min = standard*hand_long
    print("mR:",hand_m_right,"mL",hand_m_left,"dR:",hand_d_right,"dL:",hand_d_left,"HM",hand_min)
    if hand_m_right>=R_max and hand_m_right<=R_min and hand_d_right>=hand_min:
      return "left"
    elif hand_m_left<=L_max and hand_m_left>=L_min and hand_d_left>=hand_min:
      return "right"
    elif hand_m_left>=R_max and hand_m_left<=R_min and hand_d_left>=hand_min:
      return "left"
    elif hand_m_right<=L_max and hand_m_right>=L_min and hand_d_right>=hand_min:
      return "right"
    else:
      return None

# 啟用姿勢偵測
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    while True:
        img_rgb = ImageGrab.grab()
        img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        #ret, img = cap.read()
        # if not ret:
        #     print("Cannot receive frame")
        #     break
        # img = cv2.resize(img,(960,540))               # 縮小尺寸，加快演算速度
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img_bgr)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(img_bgr,results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
          for i,lm in enumerate(results.pose_landmarks.landmark):
            #print(i," x:",lm.x," y:",lm.y)
            cdn_x[i] = lm.x 
            cdn_y[i] = lm.y
        else:
          for i in range(33):
            cdn_x[i] = 0
            cdn_y[i] = 0
        
        command = tch_follower(cdn_x, cdn_y)
        if command == "right":
          ser.write(b"right")
          a = ser.readline()
          print(a)
        elif command == "left":
          ser.write(b"left")
          a = ser.readline()
          print(a)
        # data = write_space(cdn_x,cdn_y)
        # print(data)
        
        cv2.imshow('oxxostudio', img_bgr)
        if cv2.waitKey(5) == ord('q') or cv2.waitKey(5) == 27:
            break     # 按下 q 鍵停止
# cap.release()
cv2.destroyAllWindows()