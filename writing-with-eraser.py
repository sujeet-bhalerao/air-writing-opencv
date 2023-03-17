import cv2
import numpy as np

# Initialize the drawing canvas
drawing = np.zeros((480, 640, 3), dtype=np.uint8)

# Start the video capture
cap = cv2.VideoCapture(0)

# Initialize the previous point variables for the white marker and blue eraser
prev_point_white = None
prev_point_blue = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the green fingertip marker
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([90, 255, 255])


    # Define the color range for the light blue eraser marker
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Create masks based on the color ranges
    mask_white = cv2.inRange(hsv, lower_color, upper_color)
    mask_white = cv2.erode(mask_white, None, iterations=2)
    mask_white = cv2.dilate(mask_white, None, iterations=2)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.erode(mask_blue, None, iterations=2)
    mask_blue = cv2.dilate(mask_blue, None, iterations=2)

    # Find contours in the white mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a white contour is found, draw on the canvas
    if contours_white:
        max_contour_white = max(contours_white, key=cv2.contourArea)
        if cv2.contourArea(max_contour_white) > 100:
            # Find the center of the white contour
            M = cv2.moments(max_contour_white)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw on the canvas
            if prev_point_white is not None:
                cv2.line(drawing, prev_point_white, (cX, cY), (0, 255, 0), 5)
            prev_point_white = (cX, cY)
        else:
            prev_point_white = None

    # Find contours in the blue mask
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a blue contour is found, erase on the canvas
    # If a blue contour is found, erase on the canvas
    if contours_blue:
        max_contour_blue = max(contours_blue, key=cv2.contourArea)
        if cv2.contourArea(max_contour_blue) > 100:
            # Find the center of the blue contour
            M = cv2.moments(max_contour_blue)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Erase on the canvas
            if prev_point_blue is not None:
                cv2.line(drawing, prev_point_blue, (cX, cY), (0, 0, 0), 10)
            prev_point_blue = (cX, cY)
        else:
            prev_point_blue = None

    # Combine the frame and the drawing canvas
    result = cv2.addWeighted(frame, 0.7, drawing, 0.3, 0)

    # Show the result
    cv2.imshow('Air Writing', result)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
