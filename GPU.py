import cv2
import csv
import numpy as np
from tqdm import tqdm
import time
from numba import jit
import cupy as cp
from multiprocessing import Pool

@jit
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
        window = window[(window[:, 0] > 4) & (window[:, 0] < 240)]
        if window.size == 0:
            continue
        pixel_averages = cp.mean(window, axis=0)
        mean_color = cp.round(pixel_averages).astype(int)
        processed_frame[y, x] = mean_color

    return processed_frame

def main():
    coordinates = []
    with open('coordinates.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            coordinates.append((int(row[1]), int(row[0])))

    video_path = 'UAV.mp4'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1920, 1080))

    window_size = 2

    args_list = [(cp.array(cap.read()[1]), coordinates, window_size) for _ in range(total_frames)]

    with Pool() as pool, tqdm(total=total_frames, unit="frame") as pbar:
        results = pool.map(process_frame, args_list)
        for processed_frame in results:
            output_video.write(cp.asnumpy(processed_frame))
            pbar.update(1)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
