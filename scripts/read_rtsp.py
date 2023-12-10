import cv2

# RTSP URL
rtsp_url = "rtsp://127.0.0.1:8554/stream"

# Open the RTSP URL
cap = cv2.VideoCapture(rtsp_url)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Release the capture
        cap.release()
        cv2.destroyAllWindows()
        break
