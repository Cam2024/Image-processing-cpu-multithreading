import os
import cv2
import csv
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def process_frame(args):
    frame, coords, window_size = args
    processed_frame = frame.copy()

    for coord in coords:
        x, y = coord
        x_start = x - window_size
        x_end = x + window_size
        y_start = y - window_size
        y_end = y + window_size
        window = processed_frame[y_start:y_end, x_start:x_end]
        n, m, _ = window.shape
        window = window.reshape(n * m, -1)
        window = [item for item in window if 4 < item[0] < 240]
        window = np.array(window)
        if window.size == 0:
            continue
        pixel_averages = np.mean(window, axis=0)
        mean_color = np.round(pixel_averages).astype(int)
        processed_frame[y, x] = mean_color

    return processed_frame

def process_video(video_path, coordinates, output_video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 1080))

    window_size = 2

    args_list = [(cap.read()[1], coordinates, window_size) for _ in range(total_frames)]

    with Pool() as pool:
        results = pool.map(process_frame, args_list)
        for processed_frame in results:
            output_video.write(processed_frame)

    cap.release()
    output_video.release()

def main():
    parent_folder_path = "Anti-UAV-RGBT/train/"  # 替换为包含UAV_N文件夹的父文件夹路径
    file_name = "visible.mp4"

    prefix = "UAV_"
    coordinates = []

    with open('ccc.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            coordinates.append((int(row[1]), int(row[0])))

    folders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f)) and f.startswith(prefix)]

    for folder_name in tqdm(folders, desc="Processing folders"):
        video_path = os.path.join(parent_folder_path, folder_name, file_name)
        output_folder = os.path.join(parent_folder_path, folder_name)  # 每个UAV_N文件夹中
        output_video_path = os.path.join(output_folder, f"{file_name.split('.')[0]}_output.mp4")

        if os.path.exists(video_path):
            process_video(video_path, coordinates, output_video_path)
        else:
            print(f"{file_name} not found in {folder_name}")

if __name__ == "__main__":
    main()
