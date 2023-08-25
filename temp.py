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
video_path = 'Anti-UAV-RGBT/train/20190925_101846_1_8/visible.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频帧总数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义保存视频的编解码器和输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('Anti-UAV-RGBT/train/20190925_101846_1_8/output.mp4', fourcc, 20.0, (1920, 1080))

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
        for i, coord in enumerate(coordinates):
            x, y = coord
            selected = coordinates[i:i + 3]
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
            # 剔除CSV文件中已有的坐标的像素
            # print(window)
            existing_pixels = []
            for a, b in selected:
                # 获取坐标(x, y)对应的像素值，并添加到existing_pixels列表中
                existing_pixels.append(frame[b, a])

            existing_pixels = np.array(existing_pixels)

            # 使用np.isin()函数找到window中与existing_pixels中任意一个元素相同的索引
            # invert=True表示找到不匹配的索引
            mask = ~np.isin(window, existing_pixels).all(axis=1)

            # 根据mask过滤出不包含existing_pixels中任意一个元素的window元素
            new_window = window[mask]
            # print(new_window)
            # window = np.array([p for p in window if not any(np.array_equal(p, ep) for ep in existing_pixels)])
            # 计算每个像素的平均值，axis=1表示按行计算
            pixel_averages = np.mean(new_window, axis=0)
            # 取整
            mean_color = np.round(pixel_averages).astype(int)

            # 检查剔除后的窗口是否为空
            if window.size == 0:
                continue

            # 计算均值
            # mean_color = tuple(map(int, np.mean(window, axis=(0, 1))))

            # mean_color = cv2.mean(window.astype(np.uint8))
            # mean_color = tuple(map(int, cv2.mean(window.astype(np.uint8))[:-1]))

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
