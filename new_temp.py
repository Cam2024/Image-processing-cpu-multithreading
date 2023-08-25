import cv2
import csv
import numpy as np
from tqdm import tqdm

# 读取CSV文件并获取坐标数据
coordinates = []
with open('ccc.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        coordinates.append((int(row[1]), int(row[0])))  # 调换x和y的位置

# 加载视频
video_path = 'Anti-UAV-RGBT/train/20190925_101846_1_6/visible.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频帧总数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义保存视频的编解码器和输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('Anti-UAV-RGBT/train/20190925_101846_1_6/output.mp4', fourcc, 20.0, (1920, 1080))

# 定义均值滤波的窗口大小
window_size = 2  # 可以尝试调整窗口大小

frame_count = 0

with tqdm(total=total_frames, unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 遍历坐标并对像素进行均值滤波
        for coord in coordinates:
            x, y = coord
            x_start = x - window_size
            x_end = x + window_size
            y_start = y - window_size
            y_end = y + window_size
            # 提取坐标周围的像素
            window = frame[y_start:y_end, x_start:x_end]
            # 获取原数组的形状
            n, m, _ = window.shape
            # 将三维数组转换成二维数组
            window = window.reshape(n * m, -1)
            # 剔除包含的坐标
            window = [item for item in window if 4 < item[0] < 240]
            window = np.array(window)
            # 检查剔除后的窗口是否为空
            if window.size == 0:
                continue
            # 计算每个像素的平均值，axis=0表示按lie计算
            pixel_averages = np.mean(window, axis=0)
            # 取整
            mean_color = np.round(pixel_averages).astype(int)
            frame[y, x] = mean_color
        # 将处理后的帧写入输出视频文件
        output_video.write(frame)

        # 显示处理进度
        frame_count += 1
        pbar.update(1)

# 释放资源
cap.release()
output_video.release()
cv2.destroyAllWindows()
