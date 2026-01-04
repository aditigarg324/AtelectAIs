import pandas as pd
import os 


csv_path ="data/raw/chest_xray_data_labels.csv"
#images_dir = "data/raw/images"

df = pd.read_csv(csv_path)
print(" ✅ Successfully loaded!")
print(" Total no.of records", len(df))

print("Columns: ")
print(df.columns.tolist())

print(" Sample Data: ")
print(df.head())

"""
missing_files = []
for img_name in df["Image Index"]:
    if not os.path.exists(os.path.join(images_dir, img_name)):
        missing_files.append(img_name)
if missing_files:
    print(f" {len(missing_files)} image files are missing! ")
else:
    print(" ✅ All image files exist! ")
"""

print("Finding Label distribution: ")
df["Finding Labels"] = df["Finding Labels"].str.split("|")
df_exploded = df.explode("Finding Labels")
print(df_exploded["Finding Labels"].value_counts().head(15))
