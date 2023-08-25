# from PIL import Image
# import csv
# import cv2
# import numpy as np


#
# # 读取图像
# image = cv2.imread('a.jpg')
#
# # 将图像转换为灰度图像
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 阈值化处理，将所有大于阈值的像素设为255（白色）
# threshold_value = 200
# _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
#
# # 获取白色像素的坐标
# white_pixel_coordinates = np.transpose(np.nonzero(binary_image))
#
# white_coordinates = []
#
# #300 < x < 1600 and
# # 打印白色像素的坐标
# for coordinate in white_pixel_coordinates:
#     x, y = coordinate
#     if 200 < x < 900:
#         white_coordinates.append(coordinate)
#     print(f"White pixel found at ({x}, {y})")
#
#
# # 保存到CSV文件
# csv_filename = 'coordinates.csv'
# with open(csv_filename, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(['X', 'Y'])  # 写入CSV文件的表头
#     for coordinate in white_coordinates:
#         x, y = coordinate
#         writer.writerow([x, y])  # 写入坐标数据




# # 创建一个黑色背景的图像
# width = 1920
# height = 1080
# image = Image.new("RGB", (width, height), "black")
#
# # 从CSV文件中读取坐标，并将其绘制为白色
# with open('coordinates.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         try:
#             x, y = map(int, row)
#             image.putpixel((y, x), (255, 255, 255))
#         except ValueError:
#             # 非整数值，使用默认值替代
#             x, y = -1, -1
#
# # 保存图像
# image.save("output.png")






#
#
# import cv2
#
# # 回调函数，用于获取鼠标点击的坐标
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Clicked at ({x}, {y})")
#
# # 读取图像
# image = cv2.imread('output.png')
#
# # 创建窗口并绑定鼠标回调函数
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', mouse_callback)
#
# while True:
#     # 在窗口中显示图像
#     cv2.imshow('image', image)
#
#     # 按下Esc键退出循环
#     if cv2.waitKey(1) == 27:
#         break
#
# # 关闭窗口
# cv2.destroyAllWindows()









#
# import cv2
# import csv
# import numpy as np
#
# # 读取CSV文件并获取坐标数据
# coordinates = []
# with open('coordinates.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         coordinates.append((int(row[1]), int(row[0])))  # 调换x和y的位置
#
# # 加载视频
# video_path = '33.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # 获取视频帧的宽度和高度
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # 定义保存视频的编解码器和输出参数
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
#
# # 定义均值滤波的窗口大小
# window_size = 3
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 复制当前帧用于绘制坐标
#     output_frame = frame.copy()
#
#     # 遍历坐标并对像素进行均值滤波
#     for coord in coordinates:
#         x, y = coord
#
#         # 计算窗口边界
#         x_start = max(0, x - window_size)
#         x_end = min(frame.shape[1] - 1, x + window_size)
#         y_start = max(0, y - window_size)
#         y_end = min(frame.shape[0] - 1, y + window_size)
#
#         # 提取坐标周围的像素
#         window = frame[y_start:y_end, x_start:x_end]
#
#         # 检查窗口是否为空
#         if window.size == 0:
#             continue
#
#         # 检查窗口中是否存在NaN值
#         if np.isnan(window).any():
#             continue
#
#         # 计算均值
#         mean_color = tuple(map(int, np.nanmean(window, axis=(0, 1))))
#
#         # 绘制均值像素点
#         cv2.circle(output_frame, (x, y), 3, mean_color, -1)
#
#         # 将处理后的帧写入输出视频文件
#         output_video.write(output_frame)
#
#         # 显示处理后的帧
#         cv2.imshow('Output', output_frame)
#
#         # 按下 'q' 键退出循环
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# # 释放资源
# cap.release()
# output_video.release()
# cv2.destroyAllWindows()


#
# import cv2
# import csv
# import numpy as np
#
# # 读取CSV文件并获取坐标数据
# coordinates = []
# with open('coordinates.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         coordinates.append((int(row[1]), int(row[0])))  # 调换x和y的位置
#
# # 加载视频
# video_path = '33.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # 获取视频帧的宽度和高度
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # 获取视频帧总数
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# # 定义保存视频的编解码器和输出参数
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
#
# # 定义均值滤波的窗口大小
# window_size = 3
#
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 复制当前帧用于绘制坐标
#     output_frame = frame.copy()
#
#     # 遍历坐标并对像素进行均值滤波
#     for coord in coordinates:
#         x, y = coord
#
#         # 计算窗口边界
#         x_start = max(0, x - window_size)
#         x_end = min(frame.shape[1] - 1, x + window_size)
#         y_start = max(0, y - window_size)
#         y_end = min(frame.shape[0] - 1, y + window_size)
#
#         # 提取坐标周围的像素
#         window = frame[y_start:y_end, x_start:x_end]
#
#         # 检查窗口是否为空
#         if window.size == 0:
#             continue
#
#         # 检查窗口中是否存在NaN值
#         if np.isnan(window).any():
#             continue
#
#         # 检查当前坐标是否已存在于CSV文件中
#         if (x, y) in coordinates:
#             # 剔除CSV文件中已有的坐标的像素
#             existing_coords = [c for c in coordinates if c == (x, y)]
#             existing_pixels = [frame[c[1], c[0]] for c in existing_coords]
#             window = np.array([p for p in window if not any(np.array_equal(p, ep) for ep in existing_pixels)])
#
#             # 检查剔除后的窗口是否为空
#             if window.size == 0:
#                 continue
#
#
#         # 计算均值
#         mean_color = tuple(map(int, np.nanmean(window, axis=(0, 1))))
#
#         # 绘制均值像素点
#         cv2.circle(output_frame, (x, y), 3, mean_color, -1)
#
#     # 将处理后的帧写入输出视频文件
#     output_video.write(output_frame)
#
#     # 显示处理进度
#     frame_count += 1
#     progress = int((frame_count / total_frames) * 100)
#     print(f'Processing video: {progress}%')
#
# # 释放资源
# cap.release()
# output_video.release()
# cv2.destroyAllWindows()








#慢但是有用的去除东西的代码

import cv2
import csv
import numpy as np

# 读取CSV文件并获取坐标数据
coordinates = []
with open('coordinates.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        coordinates.append((int(row[1]), int(row[0])))  # 调换x和y的位置

# 加载视频
video_path = 'Anti-UAV-RGBT/train/20190925_101846_1_1/visible.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取视频帧总数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义保存视频的编解码器和输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('Anti-UAV-RGBT/train/20190925_101846_1_1/output.mp4', fourcc, 20.0, (frame_width, frame_height))

# 定义均值滤波的窗口大小
window_size = 3  # 可以尝试调整窗口大小

frame_count = 0
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
    progress = int((frame_count / total_frames) * 100)
    print(f'Processing video: {progress}%')

# 释放资源
cap.release()
output_video.release()
cv2.destroyAllWindows()

