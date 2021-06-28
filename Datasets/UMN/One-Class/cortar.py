import cv2

corte = 359
it = 0

# Input and first frame
cap = cv2.VideoCapture("UMN11.mp4")
video_open, frame = cap.read()

# Output
normal = cv2.VideoWriter("UMN11_normal.avi",cv2.VideoWriter_fourcc(*"MJPG"),30,
                         (int(cap.get(3)),int(cap.get(4))))
anomaly = cv2.VideoWriter("UMN11_anomaly.avi",cv2.VideoWriter_fourcc(*"MJPG"),30,
                        (int(cap.get(3)),int(cap.get(4))))

while video_open:
    # Write frame
    if it <= corte:
        normal.write(frame)
    else:
        anomaly.write(frame)

    # Show frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Next frame
    video_open, frame = cap.read()
    it += 1

normal.release()
anomaly.release()
