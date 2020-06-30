import cv2
import numpy as np

if __name__ == '__main__':
    src = 'tests/data/master_4s_1080.mp4'
    cap = cv2.VideoCapture(src)
    # NOTE: H264 won't work for pre-built opencv, change to mp4v
    writer = cv2.VideoWriter('rend2_4s_1080_adv_attack.mp4', cv2.VideoWriter_fourcc(*'H264'), 60, (1920, 1080))
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        frame = frame[..., ::-1]
        frame[..., 0] = np.max(frame, axis=2)
        frame[..., 1:] = 0
        writer.write(frame)
    cap.release()
    writer.release()
