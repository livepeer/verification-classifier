import cv2

cap=cv2.VideoCapture('../stream/sources/vimeo/104681073-no-audio.mp4', cv2.CAP_FFMPEG)

# cap=cv2.VideoCapture('../stream/1080_14/104681073-no-audio.mp4', cv2.CAP_FFMPEG)
# cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_BACKEND))

i=0
while cap.isOpened():
    ret, frame = cap.read()
    # cv2.imwrite('{}.jpeg'.format(i),frame)
    i+=1
    if not ret:
        break

print(i)
print(cap.get(cv2.CAP_PROP_POS_MSEC))
print(cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1))
print(cap.get(cv2.CAP_PROP_POS_MSEC))