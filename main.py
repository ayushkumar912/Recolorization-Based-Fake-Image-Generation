import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Directories
real_dir = "real_images"
gray_dir = "grayscale_images"
fake_dir = "fake_colorized_images"
os.makedirs(gray_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Load Colorization Model
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
points = "models/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.transpose().reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Convert to Grayscale
def convert_to_grayscale(img_path, save_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path, gray_bgr)

# Colorize Image (adapted for 150x150 images)
def colorize(img_path, save_path):
    orig_img = cv2.imread(img_path)
    lab_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB)
    L_channel = lab_img[:, :, 0].astype("float32")
    
    L_resized = cv2.resize(L_channel, (224, 224))
    L_resized_norm = L_resized - 50
    blob = cv2.dnn.blobFromImage(L_resized_norm)
    net.setInput(blob)
    
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channels = cv2.resize(ab_channels, (orig_img.shape[1], orig_img.shape[0]))
    ab_channels = ab_channels * 128

    lab_output = np.concatenate((L_channel[:, :, np.newaxis], ab_channels), axis=2)
    lab_output = np.clip(lab_output, 0, 255).astype("uint8")
    colorized = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)

    # Apply warm tone correction here
    colorized = correct_colors(colorized)

    cv2.imwrite(save_path, colorized)

# Color correction function (warm shift)
def correct_colors(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    la, a, b = cv2.split(lab)
    # Shift A and B channels slightly to add warmth
    a = cv2.add(a, 1)  # Adjust as needed (try 10-15)
    b = cv2.add(b, 15)  # Adjust as needed (try 15-20)
    lab = cv2.merge((la, a, b))
    final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return final

# Compute SSIM between two images
def compute_ssim(real_img, fake_img):
    real = cv2.imread(real_img)
    fake = cv2.imread(fake_img)
    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
    return ssim(real_gray, fake_gray)

# Process Images
data = []
for img_name in os.listdir(real_dir):
    real_path = os.path.join(real_dir, img_name)
    gray_path = os.path.join(gray_dir, img_name)
    fake_path = os.path.join(fake_dir, img_name)

    convert_to_grayscale(real_path, gray_path)
    colorize(gray_path, fake_path)
    
    score = compute_ssim(real_path, fake_path)
    data.append([real_path, fake_path, score])
    print(f"Processed: {img_name} | SSIM: {score:.4f}")

# Save CSV with image paths and SSIM scores
df = pd.DataFrame(data, columns=["Real Image", "Fake Image", "SSIM Score"])
df.to_csv("image_pairs.csv", index=False)
print("Dataset generation completed. CSV saved.")
