import os
import random
import shutil
import glob
import sys

# --- CONFIGURATION: EDIT THESE 3 LINES FOR EACH RUN ---

# 1. Source: REPLACE THIS with the absolute path to your downloaded folder 
# (e.g., C:\Users\YourName\Downloads\crack_dataset\Positive)
SOURCE_DIR = r"C:\Users\aakas\Downloads\archive\Negative" 

# 2. Destination: Set the target folder inside your project
# Example targets: data/images/train/Positive, data/images/val/Positive, etc.
TRAIN_DESTINATION_DIR = r"C:\Users\aakas\OneDrive\Desktop\Bridge_Health_AI\data\images\val\Negative" 

# 3. Count: Set the number of images to move (16000 for train, 4000 for val)
TRAIN_COUNT = 4000 

# --- END CONFIGURATION ---

def split_and_move_images():
    # Helper function to get the absolute path for the destination (relative to the script)
    absolute_dest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRAIN_DESTINATION_DIR)
    
    print(f"Starting split for folder: {os.path.basename(SOURCE_DIR)}")
    print(f"Moving {TRAIN_COUNT} files to: {absolute_dest_path}")

    # Get a list of all image file paths (assuming .jpg format)
    all_files = glob.glob(os.path.join(SOURCE_DIR, "*.jpg"))
    
    if not all_files:
        print(f"🚨 ERROR: No images found in the source directory: {SOURCE_DIR}")
        print("Please check the 'SOURCE_DIR' path in your configuration.")
        sys.exit()

    if len(all_files) < TRAIN_COUNT:
        print(f"🚨 ERROR: Found {len(all_files)} files, but target count is {TRAIN_COUNT}.")
        print("Cannot move more files than exist.")
        sys.exit()

    # Randomly shuffle the list of files
    random.shuffle(all_files)
    
    # Select the target number of files for the training set
    target_files = all_files[:TRAIN_COUNT]
    
    # Create the destination directory if it doesn't exist
    os.makedirs(absolute_dest_path, exist_ok=True)
    
    print(f"Total files found: {len(all_files)}")
    print(f"Moving {len(target_files)} files...")
    
    # Move the selected files to the destination
    for file_path in target_files:
        filename = os.path.basename(file_path)
        # Use shutil.move to move the file from SOURCE_DIR to the DESTINATION_DIR
        try:
            shutil.move(file_path, os.path.join(absolute_dest_path, filename))
        except Exception as e:
            print(f"Could not move file {filename}. Error: {e}")
            
    # Check remaining files in the original source folder
    remaining_files = glob.glob(os.path.join(SOURCE_DIR, "*.jpg"))
    
    print(f"✅ Move complete. Files saved to: {absolute_dest_path}")
    print(f"Remaining {len(remaining_files)} files in the original source folder.")

if __name__ == "__main__":
    split_and_move_images()