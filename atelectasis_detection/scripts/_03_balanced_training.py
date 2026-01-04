import pandas as pd 
import os
import shutil
from tqdm import tqdm 


csv_path = "data/raw/chest_xray_data_labels.csv"
training = "data/raw/train_val_list.txt"
testing = "data/raw/test_list.txt"
images_dir = "data/raw/images/"


output_dir = "data/processed/" 
os.makedirs(output_dir, exist_ok=True)

before_balancing_training_dir = os.path.join(output_dir, "before_balancing")
after_balancing_training_dir = os.path.join(output_dir, "after_balancing")
testing_dir = os.path.join(output_dir, "testing_two_labels_only")

def copy_images(df, base_dir):
    for _, row in tqdm(df.iterrows(), total = len(df), desc =f"Copying to {os.path.basename(base_dir)}"):
        label = row["Finding Labels"]
        img_name = row["Image Index"]
        destination_dir = os.path.join(base_dir, label)
        os.makedirs(destination_dir, exist_ok = True)
        source = os.path.join(images_dir, img_name)
        destination = os.path.join(destination_dir, img_name)
        if os.path.exists(source):
            shutil.copy(source, destination)

df = pd.read_csv(csv_path)
print("Loaded Successfully!")
print(f"Total recoreds: {len(df)}")

df["Finding Labels"] = df["Finding Labels"].str.split('|')
df_exploded = df.explode("Finding Labels")

with open(training, 'r') as f:
    training_files = set(f.read().splitlines())
with open(testing, 'r') as f:
    testing_files = set(f.read().splitlines())


df_training = df_exploded[(df_exploded["Image Index"].isin(training_files)) & \
                          (df_exploded["Finding Labels"].isin(["Atelectasis", "No Finding"]))]
df_testing = df_exploded[(df_exploded["Image Index"].isin(testing_files)) & \
                         (df_exploded["Finding Labels"].isin(["Atelectasis", "No Finding"]))]  


print("Finding label distribution before balancing the training images: ")
print(df_training["Finding Labels"].value_counts())
print("Finding label distribution before balancing for testing:")
print(df_testing["Finding Labels"].value_counts()) 

print(" Copying original training images: ")
copy_images(df_training, before_balancing_training_dir)
print(f" ✅ Original training images copied to {before_balancing_training_dir}") 

df_pneumonia = df_training[df_training["Finding Labels"]=="Atelectasis"]
df_nofinding = df_training[df_training["Finding Labels"]=="No Finding"].sample(n=len(df_pneumonia), random_state = 42)
df_training_images_balanced = pd.concat([df_pneumonia, df_nofinding]).sample(frac = 1, random_state = 42).reset_index(drop=True)

print("Finding label distribution after balancing the training images: ")
print(df_training_images_balanced["Finding Labels"].value_counts())

print("Copying training images after balancing: ")
copy_images(df_training_images_balanced, after_balancing_training_dir)
print(f" ✅ Balanced training images copied to {after_balancing_training_dir}")

print("Copying testing images: ")
copy_images(df_testing, testing_dir)
print(f"testing Images copied to {testing_dir}") 

print(" 🎊 images are organised! ")




