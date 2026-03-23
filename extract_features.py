import cv2
import numpy as np
import os
import pandas as pd

def extract_color_stats(image):
    # Split channels
    (B, G, R) = cv2.split(image)
    
    # 1. Fire Pixel Ratio (The most important new feature)
    # Fire logic: Red > 150 AND Red > Green AND Red > Blue
    fire_mask = (R > 150) & (R > G) & (R > B)
    fire_pixel_count = np.sum(fire_mask)
    total_pixels = image.shape[0] * image.shape[1]
    fire_ratio = fire_pixel_count / total_pixels
    
    # 2. Standard stats
    return [fire_ratio, np.mean(R), np.mean(G), np.mean(B), np.std(R)]

def extract_texture_stats(image):
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Canny Edge Detection to find "busy" areas (fire is chaotic)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
    return [edge_density]

def process_folder(folder_path, label):
    data = []
    print(f"Processing folder: {folder_path}...")
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to speed up processing (optional but recommended)
                img = cv2.resize(img, (128, 128))
                
                # Extract Features
                color_features = extract_color_stats(img)
                texture_features = extract_texture_stats(img)
                
                # Combine all features + Label (1 for Fire, 0 for No Fire)
                row = color_features + texture_features + [label]
                data.append(row)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return data

# --- MAIN EXECUTION ---
# Update these paths to match your folder structure
fire_dir = "dataset/fire_images"
non_fire_dir = "dataset/non_fire_images"

# 1 = Fire, 0 = Non-Fire
fire_data = process_folder(fire_dir, 1)
non_fire_data = process_folder(non_fire_dir, 0)

# Create DataFrame
# ... (paste the new extract_color_stats function here) ...

# ... inside the main execution block at the bottom ...
columns = ['Fire_Ratio', 'Mean_R', 'Mean_G', 'Mean_B', 'Std_R', 'Edge_Density', 'Label']
# Note: I removed Std_G and Std_B to keep it simple, make sure your return statement matches!
df = pd.DataFrame(fire_data + non_fire_data, columns=columns)

# Shuffle and Save
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("forest_fire_features.csv", index=False)
print("Success! Features saved to 'forest_fire_features.csv'")