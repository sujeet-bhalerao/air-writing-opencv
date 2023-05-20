import cv2
import numpy as np

# Start the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video capture")
    exit(0)

def nothing(x):
    pass

cv2.namedWindow('HSV Trackbars')
cv2.createTrackbar('Low H', 'HSV Trackbars', 30, 180, nothing)
cv2.createTrackbar('High H', 'HSV Trackbars', 85, 180, nothing)
cv2.createTrackbar('Low S', 'HSV Trackbars', 150, 255, nothing)
cv2.createTrackbar('High S', 'HSV Trackbars', 255, 255, nothing)
cv2.createTrackbar('Low V', 'HSV Trackbars', 50, 255, nothing)
cv2.createTrackbar('High V', 'HSV Trackbars', 255, 255, nothing)

drawing_path = []

ret, frame = cap.read()
if not ret:
    print("Error reading frame")
    exit(0)

frame = cv2.flip(frame, 1)
drawing = np.zeros_like(frame)  # Initialize the drawing canvas based on the frame size

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    frame = cv2.flip(frame, 1)
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions and update the color range
    l_h = cv2.getTrackbarPos('Low H', 'HSV Trackbars')
    h_h = cv2.getTrackbarPos('High H', 'HSV Trackbars')
    l_s = cv2.getTrackbarPos('Low S', 'HSV Trackbars')
    h_s = cv2.getTrackbarPos('High S', 'HSV Trackbars')
    l_v = cv2.getTrackbarPos('Low V', 'HSV Trackbars')
    h_v = cv2.getTrackbarPos('High V', 'HSV Trackbars')

    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([h_h, h_s, h_v])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 100:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:  # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                drawing_path.append((cX, cY))

    for point in drawing_path:
        cv2.circle(drawing, point, 5, (0, 255, 0), -1)

    result = cv2.addWeighted(frame, 0.7, drawing, 0.3, 0)
    cv2.imshow('Air Writing', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('air_drawing.png', drawing)
    elif key == ord('c'):
        drawing = np.zeros_like(frame)
        drawing_path.clear()

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Start the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video capture")
    exit(0)

def nothing(x):
    pass

cv2.namedWindow('HSV Trackbars')
cv2.createTrackbar('Low H', 'HSV Trackbars', 30, 180, nothing)
cv2.createTrackbar('High H', 'HSV Trackbars', 85, 180, nothing)
cv2.createTrackbar('Low S', 'HSV Trackbars', 150, 255, nothing)
cv2.createTrackbar('High S', 'HSV Trackbars', 255, 255, nothing)
cv2.createTrackbar('Low V', 'HSV Trackbars', 50, 255, nothing)
cv2.createTrackbar('High V', 'HSV Trackbars', 255, 255, nothing)

drawing_path = []

ret, frame = cap.read()
if not ret:
    print("Error reading frame")
    exit(0)

frame = cv2.flip(frame, 1)
drawing = np.zeros_like(frame)  # Initialize the drawing canvas based on the frame size

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    frame = cv2.flip(frame, 1)
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions and update the color range
    l_h = cv2.getTrackbarPos('Low H', 'HSV Trackbars')
    h_h = cv2.getTrackbarPos('High H', 'HSV Trackbars')
    l_s = cv2.getTrackbarPos('Low S', 'HSV Trackbars')
    h_s = cv2.getTrackbarPos('High S', 'HSV Trackbars')
    l_v = cv2.getTrackbarPos('Low V', 'HSV Trackbars')
    h_v = cv2.getTrackbarPos('High V', 'HSV Trackbars')

    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([h_h, h_s, h_v])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 100:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:  # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                drawing_path.append((cX, cY))

    for point in drawing_path:
        cv2.circle(drawing, point, 5, (0, 255, 0), -1)

    result = cv2.addWeighted(frame, 0.7, drawing, 0.3, 0)
    cv2.imshow('Air Writing', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('air_drawing.png', drawing)
    elif key == ord('c'):
        drawing = np.zeros_like(frame)
        drawing_path.clear()

cap.release()
cv2.destroyAllWindows()
