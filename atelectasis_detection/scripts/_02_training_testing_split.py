import pandas as pd 
import os 
import shutil 
from tqdm import tqdm 
csv_path = "data/raw/chest_xray_data_labels.csv" 
training_list = "data/raw/train_val_list.txt" 
testing_list = "data/raw/test_list.txt" 
images_dir = "data/raw/images/" 
output_dir = "data/processed/" 

df = pd.read_csv(csv_path) 
print("Loaded Successfully!") 
print(f"Total recoreds: {len(df)}") 

df["Finding Labels"] = df["Finding Labels"].str.split('|') 
df_exploded = df.explode("Finding Labels") 

with open(training_list, 'r') as f: 
    training_files = set(f.read().splitlines())
with open(testing_list, 'r') as f:
    testing_files = set(f.read().splitlines())

df_training_images = df_exploded[df_exploded["Image Index"].isin(training_files)] 
df_testing_images = df_exploded[df_exploded["Image Index"].isin(testing_files)] 

training_dir = os.path.join(output_dir, "training/") 
testing_dir = os.path.join(output_dir, "testing") 
os.makedirs(training_dir, exist_ok = True) 
os.makedirs(testing_dir, exist_ok = True) 

print(f" ✅ Training images: {len(df_training_images['Image Index'].unique())}") 
print(f" ✅ Testing images: {len(df_testing_images['Image Index'].unique())}") 

print("Finding label distribution in training set: ") 
print(df_training_images["Finding Labels"].value_counts().head(15)) 
print("Finding label distribution in testing set: ") 
print(df_testing_images["Finding Labels"].value_counts().head(15)) 

print("Copying training images to train folder") 
for img_name in tqdm(df_training_images["Image Index"].unique(), desc = "Training"): 
    source = os.path.join(images_dir, img_name) 
    destination = os.path.join(training_dir, img_name) 
    if os.path.exists(source): 
        shutil.copy(source, destination) 

print("Copying testing images to test folder") 
for img_name in tqdm(df_testing_images["Image Index"].unique(), desc = "Testing"): 
    source = os.path.join(images_dir, img_name) 
    destination = os.path.join(testing_dir, img_name) 
    if os.path.exists(source): 
        shutil.copy(source, destination) 
        
print("All images are successfully separated into: ") 
print(f" - {training_dir}") 
print(f" - {testing_dir}")