import cv2
import csv
import numpy as np
from tqdm import tqdm

# 读取CSV文件并获取坐标数据
coordinates = []
with open('coordinates.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        coordinates.append((int(row[1]), int(row[0])))  # 调换x和y的位置

# 加载视频
video_path = 'Anti-UAV-RGBT/train/20190925_101846_1_7/visible.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取视频帧总数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义保存视频的编解码器和输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('Anti-UAV-RGBT/train/20190925_101846_1_7/output.mp4', fourcc, 20.0, (frame_width, frame_height))

# 定义均值滤波的窗口大小
window_size = 3  # 可以尝试调整窗口大小

frame_count = 0

with tqdm(total=total_frames, unit="frame") as pbar:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 复制当前帧用于绘制坐标
        output_frame = frame.copy()

        # 遍历坐标并对像素进行均值滤波
        for coord in coordinates:
            x, y = coord

            # 计算窗口边界
            x_start = max(0, x - window_size)
            x_end = min(frame.shape[1] - 1, x + window_size)
            y_start = max(0, y - window_size)
            y_end = min(frame.shape[0] - 1, y + window_size)

            # 提取坐标周围的像素
            window = frame[y_start:y_end, x_start:x_end]

            # 检查窗口是否为空
            if window.size == 0:
                continue

            # 检查当前坐标是否已存在于CSV文件中
            if (x, y) in coordinates:
                # 剔除CSV文件中已有的坐标的像素
                existing_coords = [c for c in coordinates if c == (x, y)]
                existing_pixels = [frame[c[1], c[0]] for c in existing_coords]
                window = np.array([p for p in window if not any(np.array_equal(p, ep) for ep in existing_pixels)])

                # 检查剔除后的窗口是否为空
                if window.size == 0:
                    continue

            # 计算均值
            mean_color = tuple(map(int, np.mean(window, axis=(0, 1))))

            # 在原始帧上用均值颜色填充白线像素
            for i in range(y_start, y_end):
                for j in range(x_start, x_end):
                    if (j, i) in coordinates:
                        frame[i, j] = mean_color

        # 将处理后的帧写入输出视频文件
        output_video.write(frame)

        # 显示处理进度
        frame_count += 1
        pbar.update(1)

# 释放资源
cap.release()
output_video.release()
cv2.destroyAllWindows()