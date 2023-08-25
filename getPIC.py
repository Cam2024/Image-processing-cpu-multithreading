import cv2

def extract_frame(video_path, frame_number):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 确保视频文件已经打开
    if not video_capture.isOpened():
        print("无法打开视频文件")
        return

    # 设置视频的位置为第20帧
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # 读取并抽取第20帧
    ret, frame = video_capture.read()

    # 确保成功读取了帧
    if not ret:
        print("无法读取帧")
        return

    # 保存第20帧为图像文件
    output_path = f"frame_{frame_number:03d}.png"
    cv2.imwrite(output_path, frame)

    # 关闭视频文件
    video_capture.release()

    print(f"第{frame_number}帧已抽取并保存为{output_path}")

# 调用函数并传入视频文件路径和要抽取的帧数
video_path = "Anti-UAV-RGBT/train/20190925_101846_1_8/visible.mp4"
frame_number_to_extract = 20
extract_frame(video_path, frame_number_to_extract)
