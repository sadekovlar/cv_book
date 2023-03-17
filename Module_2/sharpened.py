import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('../data/processing/trm.168.091.avi')

# Define the sharpening kernel
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# Loop through each frame of the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
    
    # If the frame was read successfully, process it
    if ret:
        # Apply the sharpening kernel to the frame
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Display the processed frame
        cv2.imshow('Input', frame)
        cv2.imshow('Sharpened Video', sharpened)
        
        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # If the frame was not read successfully, exit the loop
    else:
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()
