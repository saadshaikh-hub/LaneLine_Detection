# lane_detection.py
# Develop by Mohd Saad Shaikh , M.sc Data Science & Big Data Analytics, Part-1 , 2024
# This script is used to detect lanes in an image using OpenCV and Hough Transform.


import cv2
import numpy as np

def region_of_interest(img, vertices):
    """Define the region of interest for lane detection."""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def process_frame(frame):
    """Process each frame for lane detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define the region of interest
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, roi_vertices)

    # Hough transform for line detection
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=5)

    # Draw the lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def main():
    """Main function to run lane detection on video."""
    cap = cv2.VideoCapture('data/test_video.mp4')  # Load the video file

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)

        # Display the result
        cv2.imshow('Lane Line Detection', processed_frame)
        # Save the processed frame
        cv2.imwrite('output/output_image.jpg', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
