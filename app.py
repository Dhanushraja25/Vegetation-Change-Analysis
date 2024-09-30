import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Define upload and output directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to normalize the image
def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Function to align images using ORB keypoints and homography
def align_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, channels = img1.shape
    img2_aligned = cv2.warpPerspective(img2, h, (width, height))

    aligned_img_path = os.path.join(OUTPUT_FOLDER, 'aligned_year2.png')
    cv2.imwrite(aligned_img_path, img2_aligned)

    return aligned_img_path

# Function to process (normalize) and display images using matplotlib
def process_and_display_images(img1_path, aligned_img_path):
    img1 = cv2.imread(img1_path)
    img2_aligned = cv2.imread(aligned_img_path)

    normalized_img1 = normalize_image(img1)
    normalized_img2 = normalize_image(img2_aligned)

    normalized_img1_path = os.path.join(OUTPUT_FOLDER, 'normalized_year1.png')
    normalized_img2_path = os.path.join(OUTPUT_FOLDER, 'normalized_year2.png')
    cv2.imwrite(normalized_img1_path, normalized_img1)
    cv2.imwrite(normalized_img2_path, normalized_img2)

    # Convert images to RGB for plotting
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_aligned_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)

    # Create and save the plot without displaying it
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title('Year 1 Image')
    plt.imshow(img1_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Aligned Year 2 Image')
    plt.imshow(img2_aligned_rgb)
    plt.axis('off')

    output_plot_path = os.path.join(OUTPUT_FOLDER, 'comparison_plot.png')
    plt.savefig(output_plot_path)  # Save the figure
    plt.close()  # Close the figure to free up memory

    return normalized_img1_path, normalized_img2_path, output_plot_path

def calculate_exg(image):
    green = image[:, :, 1].astype(float)
    red = image[:, :, 0].astype(float)
    blue = image[:, :, 2].astype(float)
    exg = 2 * green - red - blue
    exg = np.clip(exg, 0, 255)
    return exg

def analyze_vegetation_change(normalized_img1_path, normalized_img2_path):
    img1 = cv2.imread(normalized_img1_path)
    img2 = cv2.imread(normalized_img2_path)

    exg1 = calculate_exg(img1)
    exg2 = calculate_exg(img2)

    difference = exg2 - exg1

    increase_threshold = 20
    decrease_threshold = -20

    increase = difference > increase_threshold
    decrease = difference < decrease_threshold

    total_pixels = difference.size
    increase_pixels = np.sum(increase)
    decrease_pixels = np.sum(decrease)

    increase_percentage = (increase_pixels / total_pixels) * 100
    decrease_percentage = (decrease_pixels / total_pixels) * 100

    return increase_percentage, decrease_percentage

def apply_overlay(image, mask, color):
    """Apply a colored overlay to the areas defined by the mask."""
    overlay = np.zeros_like(image)  # Create an empty overlay
    overlay[mask] = color  # Set the overlay color where the mask is True
    combined = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)  # Blend the images
    return combined

def generate_overlays(normalized_img1_path, normalized_img2_path):
    img1 = cv2.imread(normalized_img1_path)
    img2 = cv2.imread(normalized_img2_path)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    exg1 = calculate_exg(img1)
    exg2 = calculate_exg(img2)

    # Calculate difference in ExG
    difference = exg2 - exg1

    # Create masks
    increase_threshold = 20
    decrease_threshold = -20

    increase = difference > increase_threshold
    decrease = difference < decrease_threshold

    # Define colors for overlays
    green_color = [0, 255, 0]  # Green for increase
    red_color = [255, 0, 0]    # Red for decrease

    # Apply overlays to the original images
    increase_overlay_img = apply_overlay(img2, increase, green_color)
    decrease_overlay_img = apply_overlay(img2, decrease, red_color)

    # Save the overlay images
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'increase_overlay.png'), cv2.cvtColor(increase_overlay_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'decrease_overlay.png'), cv2.cvtColor(decrease_overlay_img, cv2.COLOR_RGB2BGR))

    return 'increase_overlay.png', 'decrease_overlay.png'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image1' not in request.files or 'image2' not in request.files:
        return "No file part", 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return "No selected file", 400

    img1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, file2.filename)

    file1.save(img1_path)
    file2.save(img2_path)

    aligned_img_path = align_images(img1_path, img2_path)

    normalized_img1_path, normalized_img2_path, output_plot_path = process_and_display_images(img1_path, aligned_img_path)

    # Analyze vegetation change
    increase_percentage, decrease_percentage = analyze_vegetation_change(normalized_img1_path, normalized_img2_path)

    # Format percentages to two decimal places
    increase_percentage = f"{increase_percentage:.2f}"
    decrease_percentage = f"{decrease_percentage:.2f}"

    # Generate overlays
    increase_overlay_img, decrease_overlay_img = generate_overlays(normalized_img1_path, normalized_img2_path)

    output_images = os.listdir(OUTPUT_FOLDER)
    output_images = [img for img in output_images if img.endswith(('.png', '.jpg', '.jpeg'))]

    return render_template('output.html', output_images=output_images, 
                           increase_percentage=increase_percentage, 
                           decrease_percentage=decrease_percentage,
                           increase_overlay_img=increase_overlay_img,
                           decrease_overlay_img=decrease_overlay_img)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
