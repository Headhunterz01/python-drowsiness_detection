# 疲劳检测系统

　　这是一个基于计算机视觉的疲劳检测系统,可以通过分析人脸的眼睛和嘴部特征,实时检测疲劳和打哈欠情况。当检测到疲劳或打哈欠时,程序会保存当前画面并显示警告信息。

## 创作目的

　　据统计，疲劳驾驶发生的交通事故占**中国**全国交通事故总量的21%，并且疲劳驾驶发生的交通事故死亡率高达83%。另一份《道路交通事故统计年报》显示，我国每年因疲劳驾驶直接引发的道路交通事故有<font color = 'red'> **1000**</font>余起，造成2000余人死亡或重伤。根据**美国**国家公路交通安全管理局的数据，每年约有 100,000 起警方报告的车祸涉及*疲劳驾驶*。这些事故造成超过<font color = 'red'> **1,550人死亡，71,000人受伤**</font>。然而，实际数字可能要高得多，因为很难确定司机在发生车祸时是否昏昏欲睡。
　　因此，我们试图建立一个系统，检测一个人是否昏昏欲睡并提醒他。

___

## 环境配置

### 依赖库

- ***Python 3.6 或更高版本***
- ***OpenCV (cv2)***
- ***Dlib***
- ***Numpy***
- ***Scipy***

### 安装和配置 Dlib

　　我们需要创建一个虚拟环境来安装 Dlib，因为它不能使用 pip 直接安装。因此，如果您以前没有安装过 Dlib，请按照以下命令将 Dlib 安装到您的系统中。确保您已安装 Anaconda，因为我们将在 Anaconda Prompt 中执行所有操作，注意，请以管理员身份运行

#### 第1步 - 更新 conda

```bash
conda update conda
```

#### 第2步 - 更新 anaconda

```bash
conda update anaconda 
```

#### 第3步 - 创建虚拟环境

```
conda create -n env_dlib 
```

#### 第4步 - 激活虚拟环境

```
conda activate env_dlib
```

#### 第5步 - 安装dlib库

```
conda install -c conda-forge Dlib 
```

如果所有这些步骤都成功完成，那么 Dlib 将安装在虚拟环境中env_dlib。请确保使用此环境来运行整个项目。

___

**停用虚拟环境命令**

```
conda deactivate 
```

**你可以使用以下命令安装所需的Python库:**

```
pip install opencv-python Dlib numpy scipy
```

### 其他依赖

　　该程序还需要dlib的预训练人脸形状预测器模型文件 `shape_predictor_68_face_landmarks.dat`。下载地址：
    https://drive.google.com/file/d/1wHZxE5TJ5KO0Dktn5ZMbUkYoXQ_w5DGx/view?usp=sharing



## 使用方法

1. 确保你已经正确配置了运行环境并下载了必要的模型文件。

2. 打开终端,切换到程序所在目录。

3. 运行程序`python drowsiness_detection.py`

4. 程序将打开你的电脑摄像头,并开始实时检测你的面部状态。

5. 如果检测到疲劳或打哈欠,程序会在视频画面上显示警告信息，
   并在 `drowsy_images` 文件夹中保存当前画面的图像文件。
   
   ![drowsy_2024-03-16_14-55-53.jpg](drowsy_2024-03-16_14-55-53.jpg)

6. 按下 `Esc` 键可以退出程序。

___

## 代码结构

├── `drowsiness_detection.py`                               <font color=red>*# 主程序文件*</font>
├── `drowsy_images`                                                    <font color=red>*# 用于存储检测到疲劳时保存的图像*</font>
└── `shape_predictor_68_face_landmarks.dat`  <font color=red>*# dlib人脸形状预测器模型文件*</font>

　　`drowsiness_detection.py` 文件包含了全部的程序代码。它使用 `OpenCV` 进行视频捕获和图像处理,使用 `Dlib` 进行人脸检测和特征点提取,并通过计算眼睛纵横比和嘴部张开程度来判断疲劳和打哈欠情况。

## 注意事项

- 该程序需要在有较好光线条件下运行,以确保人脸检测的准确性。
- 程序默认使用电脑的内置摄像头,如果需要使用其他摄像头：
  请修改 `cap = cv2.VideoCapture(0)` 这一行中的参数。
- 你可以根据需要调整程序中的各种阈值参数,如 `EAR_THRESHOLD`、`EAR_ALARM_THRESHOLD`、`EAR_CONSEC_FRAMES` 和 `YAWN_ALARM_THRESHOLD`等。

# POWER BY 顺德区LJZX
