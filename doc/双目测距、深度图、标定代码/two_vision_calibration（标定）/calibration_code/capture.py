import cv2
import os

cap = cv2.VideoCapture(1)

# set the video frame width and height
cap.set(3,1280)
cap.set(4,480)

if not os.path.exists("left"):
    os.makedirs("left")

if not os.path.exists("right"):
    os.makedirs("right")

i = 1
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # split the frame into left and right
        left_frame = frame[:, :640, :]
        right_frame = frame[:, 640:, :]

        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            left_filename = "left" + str(i) + ".png"
            right_filename = "right" + str(i) + ".png"
            cv2.imwrite("left/" + left_filename, left_frame)
            cv2.imwrite("right/" + right_filename, right_frame)
            print("Image saved! left image:", left_filename)
            print("Image saved! right image:", right_filename)
            i += 1

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    cap.release()
