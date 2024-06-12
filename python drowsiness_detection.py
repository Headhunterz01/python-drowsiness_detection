# 导入必要的模块
import cv2 # OpenCV用于计算机视觉任务
import dlib # Dlib库用于人脸检测和关键点定位
import numpy as np # NumPy用于数值计算
import os # OS模块用于与操作系统交互,如文件夹操作
from datetime import datetime # datetime模块用于获取当前时间
from scipy.spatial import distance # SciPy的spatial模块用于计算欧几里得距离

# 定义一个函数来计算眼睛长宽比值(Eye Aspect Ratio, EAR)
# 这个比值可以用来判断眼睛是睁开还是闭合
def eye_aspect_ratio(eye):
    # 计算眼睛两个垂直边的欧几里得距离
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # 计算眼睛水平边的欧几里得距离
    C = distance.euclidean(eye[0], eye[3])    
    # 如果任何一条边的距离为0,则返回None
    if A == 0.0 or B == 0.0 or C == 0.0:
        return None 
    ear = (A + B) / (2.0 * C) # 计算EAR
    return ear
# 定义一个函数来计算嘴部张开比值(Mouth Open Ratio, MOR)
# 这个比值可以用来检测是否打了哈欠
def mouth_open_ratio(mouth):
    # 计算嘴部的宽度
    mouth_width = distance.euclidean(mouth[0], mouth[6])    
    # 计算嘴部的高度
    mouth_height = distance.euclidean(mouth[3], mouth[9])
    # 如果宽度为0,返回0
    if mouth_width == 0:
        return 0
    else:# 计算MOR
        return mouth_height / mouth_width

# 初始化人脸检测器和形状预测器
detector = dlib.get_frontal_face_detector() # Dlib的人脸检测器
predictor = dlib.shape_predictor("D:/Desktop/workspace/contest/shape_predictor_68_face_landmarks.dat") # Dlib的形状预测器,用于检测面部68个关键点

# 设置阈值
EAR_THRESHOLD = 0.3 # EAR阈值,低于此值表示眼睛没有完全睁开
EAR_ALARM_THRESHOLD = 0.15 # EAR疲劳警告阈值,低于此值表示可能疲劳
EAR_CONSEC_FRAMES = 3 # 连续多少帧低于EAR阈值才算作一次眨眼
YAWN_ALARM_THRESHOLD = 0.7 # MOR打哈欠警告阈值,高于此值表示可能打哈欠

# 初始化计数器
frame_counter = 0 # 用于眨眼计数的帧计数器
blink_counter = 0 # 眨眼次数
yawn_counter = 0 # 打哈欠次数

# 获取眼睛和嘴部的landmark索引范围
lStart, lEnd = 36, 42 # 左眼landmark索引范围
rStart, rEnd = 42, 48 # 右眼landmark索引范围  
mStart, mEnd = 48, 68 # 嘴部landmark索引范围

# 设置保存图片的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录
drowsy_folder = os.path.join(current_dir, "drowsy_images") # 在当前目录下创建drowsy_images文件夹
if not os.path.exists(drowsy_folder): # 如果该文件夹不存在
    os.makedirs(drowsy_folder) # 创建该文件夹

# 打开默认摄像头
cap = cv2.VideoCapture(0)

# 初始化一些标志位
prev_yawn = False # 上一帧是否检测到打哈欠
prev_ear_alarm = False # 上一帧是否触发了疲劳警告
prev_ear = 1.0 # 上一帧的EAR值

# 开始循环处理每一帧
while True:
    ret, frame = cap.read() # 从摄像头读取一帧
    if not ret: # 如果没有成功读取帧
        break # 退出循环
    
    # 将BGR格式的彩色帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 在灰度图像上检测人脸
    faces = detector(gray)
    
    # 如果检测到多个人脸,选择最大的那个
    max_face = None
    max_area = 0
    for face in faces:
        area = (face.right() - face.left()) * (face.bottom() - face.top())
        if area > max_area:
            max_area = area
            max_face = face
    
    # 如果检测到至少一个人脸        
    if max_face is not None:
        # 在原始帧上绘制人脸边界框
        x1, y1, x2, y2 = max_face.left(), max_face.top(), max_face.right(), max_face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 使用形状预测器获取68个面部关键点
        shape = predictor(gray, max_face)
        shape_np = np.array([(p.x, p.y) for p in shape.parts()])
        
        # 从68个关键点中提取眼睛和嘴部的坐标
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        mouth = shape_np[mStart:mEnd]
        
        # 计算左右眼睛的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # 计算嘴部的MOR
        mouth_ratio = mouth_open_ratio(mouth)
        
        # 如果无法计算左右眼睛的EAR,跳过当前帧
        if leftEAR is None or rightEAR is None:
            continue
        
        # 计算两只眼睛的平均EAR
        ear = (leftEAR + rightEAR) / 2.0
        
        # 在原始帧上绘制眼睛和嘴部的边界框
        left_eye_rect = cv2.boundingRect(np.array([leftEye]))
        right_eye_rect = cv2.boundingRect(np.array([rightEye]))
        mouth_rect = cv2.boundingRect(np.array([mouth]))
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (mouth_rect[0], mouth_rect[1]), (mouth_rect[0] + mouth_rect[2], mouth_rect[1] + mouth_rect[3]), (0, 255, 0), 2)
        
        # 如果平均EAR低于阈值,增加帧计数器
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            # 如果连续EAR_CONSEC_FRAMES帧低于阈值,则认为发生了一次眨眼
            if frame_counter >= EAR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0 # 重置帧计数器
            
        save_drowsy_image = False # 标记是否需要保存当前帧
        
        # 如果嘴部张开比值高于阈值,并且上一帧没有检测到打哈欠
        if mouth_ratio > YAWN_ALARM_THRESHOLD and not prev_yawn:
            yawn_counter += 1 # 增加打哈欠计数器
            prev_yawn = True # 标记已检测到打哈欠
            save_drowsy_image = True # 标记需要保存当前帧
            print("Yawn detected!") # 打印提示
        # 如果嘴部张开比值仍然高于阈值
        elif mouth_ratio > YAWN_ALARM_THRESHOLD:
            cv2.putText(frame, "WARNING!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # 在原始帧上绘制警告文本
        else:
            prev_yawn = False # 重置打哈欠标记
            
        # 如果平均EAR低于疲劳阈值,并且上一帧未触发疲劳警告    
        if ear < EAR_ALARM_THRESHOLD and prev_ear >= EAR_ALARM_THRESHOLD:
            save_drowsy_image = True # 标记需要保存当前帧
            print("Fatigue detected!") # 打印提示
            
        # 如果需要保存当前帧    
        if save_drowsy_image:
            now = datetime.now() # 获取当前时间
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") # 格式化时间戳
            image_name = f"drowsy_{timestamp}.jpg" # 构造图像文件名
            image_path = os.path.join(drowsy_folder, image_name) # 获取图像文件完整路径
            cv2.imwrite(image_path, frame) # 保存当前帧
            print(f"Drowsy image saved: {image_path}") # 打印提示
            
        # 在原始帧上绘制状态信息
        cv2.putText(frame, "Blinks: {}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 250), 1)
        cv2.putText(frame, "Yawns: {}".format(yawn_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 如果平均EAR低于疲劳阈值
        if ear < EAR_ALARM_THRESHOLD:
            # 并且上一帧未触发疲劳警告
            if not prev_ear_alarm:
                print("WARNING! Fatigue detected.") # 打印提示
                prev_ear_alarm = True # 标记已触发疲劳警告
            cv2.putText(frame, "WARNING!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # 在原始帧上绘制警告文本
        else:
            prev_ear_alarm = False # 重置疲劳警告标记
            
        prev_ear = ear # 记录当前EAR值
        
    cv2.imshow("Frame", frame) # 显示处理后的帧
    
    key = cv2.waitKey(1) & 0xFF # 获取键盘输入
    if key == 27: # 如果按下ESC键
        break # 退出循环
        
# 释放资源        
cap.release()
cv2.destroyAllWindows()