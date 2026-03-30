import cv2
import os

DATA_DIR = './data'
label = input("Enter label (A-Z): ")
save_dir = os.path.join(DATA_DIR, label)

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f'Images: {count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collect Images - Press S to Save, Q to Quit", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f'{save_dir}/{count}.jpg', frame)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
