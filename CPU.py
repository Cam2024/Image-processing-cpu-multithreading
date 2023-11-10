import cv2
import csv
import numpy as np
from multiprocessing import Pool

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

    args_list = [(cap.read()[1], coordinates, window_size) for _ in range(total_frames)]

    with Pool() as pool:
        results = pool.map(process_frame, args_list)
        for processed_frame in results:
            output_video.write(processed_frame)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
