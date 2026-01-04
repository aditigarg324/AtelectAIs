import os
import pandas as pd
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

print("loading dataset info...")
df = pd.read_csv(csv_path)
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

df_pneumonia = df_training[df_training["Finding Labels"]=="Atelectasis"]
df_nofinding = df_training[df_training["Finding Labels"]=="No Finding"].sample(n=len(df_pneumonia), random_state = 42)
df_training_images_balanced = pd.concat([df_pneumonia, df_nofinding]).sample(frac = 1, random_state = 42).reset_index(drop=True)


for folder in [after_balancing_training_dir, before_balancing_training_dir, testing_dir]:
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        print("Please ensure that the balanced script has been run before verification.")
        exit(1)

confirm = input("\nVerification will read all image files - ensuring no copy is running in parallel. Continue (y/n): ")
if confirm.lower() != 'y':
    print("Verification aborted.")
    exit(0)

def count_images_per_label(base_dir):
    counts ={}
    for label in os.listdir(base_dir):
        folder = os.path.join(base_dir, label)
        if os.path.isdir(folder):
            counts[label] = len([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpeg", ".jpg"))])
    return counts

def check_missing_per_label(df, base_dir):
    missing_summary  ={}
    for label in df["Finding Labels"].unique():
        label_df = df[df["Finding Labels"] == label]
        folder = os.path.join(base_dir, label)
        if not os.path.exists(folder):
            missing_summary[label] = len(label_df)
            continue

        existing_files = set(os.listdir(folder))
        missing_files = [img for img in label_df["Image Index"] if img not in existing_files]
        missing_summary[label] = len(missing_files)
    return missing_summary 

print("\n Verifying before balancing folder")
missing_before = check_missing_per_label(df_training, before_balancing_training_dir)
counts_before = count_images_per_label(before_balancing_training_dir)

print("✅ Before balancing folder summary:")
for label in ["Atelectasis", "No Finding"]:
    print(f" - {label}: expected={len(df_training[df_training['Finding Labels']==label])}, "
          f"found={counts_before.get(label,0)}, missing={missing_before.get(label,0)}")

print("\nVerifying after_balancing folder...")
missing_after = check_missing_per_label(df_training_images_balanced, after_balancing_training_dir)
counts_after = count_images_per_label(after_balancing_training_dir)

print("✅ After balancing folder summary:")
for label in ["Atelectasis", "No Finding"]:
    print(f" - {label}: expected={len(df_training_images_balanced[df_training_images_balanced['Finding Labels']==label])}, "
          f"found={counts_after.get(label,0)}, missing={missing_after.get(label,0)}")

print("\nVerifying testing folder...")
missing_test = check_missing_per_label(df_testing, testing_dir)
counts_test = count_images_per_label(testing_dir)

print("✅ Testing folder summary:")
for label in ["Atelectasis", "No Finding"]:
    print(f" - {label}: expected={len(df_testing[df_testing['Finding Labels']==label])}, "
          f"found={counts_test.get(label,0)}, missing={missing_test.get(label,0)}")

print("🎯Verification Complete")