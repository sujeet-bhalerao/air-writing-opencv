import cv2
import numpy as np

# Initialize the drawing canvas
drawing = np.zeros((480, 640, 3), dtype=np.uint8)

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the fingertip marker
    lower_color = np.array([30, 150, 50])
    upper_color = np.array([85, 255, 255])

    # Create a mask based on the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found, draw on the canvas
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 100:
            # Find the center of the contour
            M = cv2.moments(max_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw on the canvas
            cv2.circle(drawing, (cX, cY), 5, (0, 255, 0), -1)

    # Combine the frame and the drawing canvas
    result = cv2.addWeighted(frame, 0.7, drawing, 0.3, 0)

    # Show the result
    cv2.imshow('Air Writing', result)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
