import cv2
import numpy as np
from .socket_server import ws_server
import asyncio

def detect_blob(np_image,save_output=False,return_output=False):
    # convert numpy image to opencv image
    gray = np_image
    # Set up the blob detector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255


    # Filter by color (black blobs on a white background)

    params.filterByColor = False
    params.blobColor = 0

    # Filter by area
    params.filterByArea = True
    params.minArea = 10  # Minimum area of blobs
    params.maxArea = 200  # Maximum area of blobs

    # Optional filters (disable for now)
    params.filterByCircularity = False
    params.filterByConvexity = False

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(gray)

    # Print point coordinates relative to the image center
    # assuming tthe image width and height are even and equal to 50mx50m in real world 
    object_list = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        angle = keypoint.angle
        x -= gray.shape[1] / 2
        y -= gray.shape[0] / 2
        x = x * 50 / gray.shape[1]
        y = y * 50 / gray.shape[0]

        # emit the blob coordinates to the connected clients the emit function is async
        # send json object with the blob coordinates
        json_obj = {
            "x": x,
            "y": y,
            "angle": angle
        }
        object_list.append(json_obj)
    dic = {
        "list": object_list
    }
 
    asyncio.run(ws_server.emit(str(dic)))


    if save_output:
        # Draw keypoints on the original image
        output_image = cv2.drawKeypoints(gray, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the output image with blobs marked
        output_path = "blob_detection_output.png"
        cv2.imwrite(output_path, output_image)

    if return_output:
        output_image = cv2.drawKeypoints(gray, keypoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return output_image 


if __name__ == "__main__":
    # Load an image
    np_image = cv2.imread("blob_detection_input.png")
    detect_blob(np_image)