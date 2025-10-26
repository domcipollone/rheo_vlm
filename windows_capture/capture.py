import cv2, os, time

save_dir = "../frames"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "latest.jpg")

cap = cv2.VideoCapture(0)  # default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    time.sleep(1)  # adjust fps