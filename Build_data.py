import cv2
import numpy as np
import os
import random
import shutil

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0
folder_name = input("Name:")  # Folder name to be added

# Create a folder to store the images
os.makedirs(f'./Images/{folder_name}', exist_ok=True)

# Collect 200 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = f'./Images/{folder_name}/{count}.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 200:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")

# Split into train and test folder
# Set the path for the original dataset
dataset_path = "Images"

# Get a list of all folders in the dataset path
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# Set the split ratio
split_ratio = 0.8  # 80% for training, 20% for testing

# Set the destination paths for train and test folders
train_destination = "Dataset/Train"
test_destination = "Dataset/Test"

# Iterate through each folder
for folder_name in folders:
    # Create the train/test subfolders in the destination paths
    train_folder = os.path.join(train_destination, folder_name)
    test_folder = os.path.join(test_destination, folder_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(os.path.join(dataset_path, folder_name)) if
                   f.endswith(".jpg") or f.endswith(".png")]

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Calculate the split index based on the split ratio
    split_index = int(len(image_files) * split_ratio)

    # Copy/move images to the train folder
    for image_file in image_files[:split_index]:
        src_path = os.path.join(dataset_path, folder_name, image_file)
        dst_path = os.path.join(train_folder, image_file)
        shutil.copy(src_path, dst_path)  # Use shutil.move if you want to move instead of copy

    # Copy/move images to the test folder
    for image_file in image_files[split_index:]:
        src_path = os.path.join(dataset_path, folder_name, image_file)
        dst_path = os.path.join(test_folder, image_file)
        shutil.copy(src_path, dst_path)  # Use shutil.move if you want to move instead of copy

print("Dataset split into train and test folders successfully!")
