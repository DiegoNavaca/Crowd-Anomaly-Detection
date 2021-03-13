import cv2

cap = cv2.VideoCapture("UMN1.mp4")

video_open, prev_frame = cap.read()

print(video_open)
cv2.imshow("",prev_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("prueba5.png",prev_frame)

cap.release()
