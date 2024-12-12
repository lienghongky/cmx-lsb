import cv2
import numpy as np

def detect_rectangles(image_path, approx_width, approx_height, tolerance=0.2):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 10, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        #visualize the contours
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 1)

        
        # Check if the approximated contour has 4 points (rectangle)
        if len(approx) == 4:
        
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Extract rectangle properties
            (center_x, center_y), (width, height), angle = rect
            angle = round(angle, 2)
            print(f"Center: ({center_x}, {center_y}), Width: {width}, Height: {height}, Angle: {angle}")
            # Check size constraints
            if (approx_width * (1 - tolerance) <= width <= approx_width * (1 + tolerance) and
                approx_height * (1 - tolerance) <= height <= approx_height * (1 + tolerance)):
                # Draw the rotated rectangle
                cv2.drawContours(image, [box], -1, (255, 0, 0), 2)
           
                # Annotate the image with the orientation
                text = f"Angle: {angle}°"
                cv2.putText(image, text, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # Draw the rotated rectangle
                cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
           

    # save the output image
    cv2.imwrite('output_image.png', image)

# Example usage
detect_rectangles('visualization/bev/output_image.png', approx_width=3, approx_height=0.5, tolerance=0.2)