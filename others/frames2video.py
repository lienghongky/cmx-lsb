import os
import cv2
from PIL import Image

# path to the Google Drive folder with images
path = "visualization"


height = 784
width = 1056

# Counting the number of images in the directory
num_of_images = len([file for file in os.listdir(path) if file.endswith((".jpg", ".jpeg", ".png"))])
print("Number of Images:", num_of_images)

# Function to generate video
def generate_video():
    image_folder = path
    video_name = 'vs1.avi'

    images = [img for img in os.listdir('.') if img.endswith((".jpg", ".jpeg", ".png"))]

    # Video writer to create .mp4 file
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

# Calling the function to generate the video
generate_video()