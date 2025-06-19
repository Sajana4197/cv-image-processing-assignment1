import cv2
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# Load image
image_path = input("Enter the path to your image: ").strip('"')
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

print("Image loaded successfully!")

# Task 1:
def reduce_levels(img, levels):
    factor = 256 // levels
    return (img // factor) * factor

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

levels = int(input("Enter desired number of intensity levels (2, 4, 8, 16, 32, 64, 128, 256): "))

current_levels = 256
while current_levels >= levels:
    reduced_image = reduce_levels(gray, current_levels)
    cv2.imwrite(f"results/reduced_{current_levels}_levels.jpg", reduced_image)
    print(f"Created image with {current_levels} intensity levels")
    current_levels = current_levels // 2

print(f"Task 1 completed: Intensity reduced from 256 to {levels} levels")

# Task 2: 
avg_3x3 = cv2.blur(image, (3, 3))
avg_10x10 = cv2.blur(image, (10, 10))
avg_20x20 = cv2.blur(image, (20, 20))

cv2.imwrite("results/average_3x3.jpg", avg_3x3)
cv2.imwrite("results/average_10x10.jpg", avg_10x10)
cv2.imwrite("results/average_20x20.jpg", avg_20x20)

print("Task 2 completed: Spatial averaging")

# Task 3: 
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, matrix, (w, h))

rotated_45 = rotate_image(image, 45)
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite("results/rotated_45.jpg", rotated_45)
cv2.imwrite("results/rotated_90.jpg", rotated_90)

print("Task 3 completed: Image rotation")

# Task 4: 
def block_average(img, block_size):
    h, w = img.shape
    result = img.copy()
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            
            block = img[y:y_end, x:x_end]
            avg_value = np.mean(block)
            
            result[y:y_end, x:x_end] = avg_value
    
    return result

block_3x3 = block_average(gray, 3)
block_5x5 = block_average(gray, 5)  
block_7x7 = block_average(gray, 7)

cv2.imwrite("results/block_3x3.jpg", block_3x3)
cv2.imwrite("results/block_5x5.jpg", block_5x5)
cv2.imwrite("results/block_7x7.jpg", block_7x7)

print("Task 4 completed: Block averaging")
print("All tasks finished! Check the 'results' folder for saved images.")