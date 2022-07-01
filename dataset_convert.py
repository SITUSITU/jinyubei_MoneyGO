import os, shutil
import glob
import pandas as pd



old_dir = "./dataset"
new_dir = "./new_dataset"



dataframe = pd.DataFrame(columns=("image_ID", "class", "label"))

ASC_H = glob.glob(os.path.join(old_dir, "*", "ASC-H&HSIL", "*"))
ASC_U = glob.glob(os.path.join(old_dir, "*", "ASC-US&LSIL", "*"))
NILM = glob.glob(os.path.join(old_dir, "*", "NILM", "*"))
SCC = glob.glob(os.path.join(old_dir, "*", "SCC&AdC", "*"))

for i, image_dir_name in enumerate(ASC_H):
    image_name = image_dir_name.split("\\")[-1]
    dataframe = dataframe.append([{"image_ID": image_name, "class": "ASC-H&HSIL", "label": int(0)}], ignore_index=True)
    print(f"ASC-H&HSIL： {i}/{len(ASC_H)}")

for i, image_dir_name in enumerate(ASC_U):
    image_name = image_dir_name.split("\\")[-1]
    dataframe = dataframe.append([{"image_ID": image_name, "class": "ASC-US&LSIL", "label": int(1)}], ignore_index=True)
    print(f"ASC-US&LSIL： {i}/{len(ASC_U)}")

for i, image_dir_name in enumerate(NILM):
    image_name = image_dir_name.split("\\")[-1]
    dataframe = dataframe.append([{"image_ID": image_name, "class": "NILM", "label": int(2)}], ignore_index=True)
    print(f"NILM： {i}/{len(NILM)}")

for i, image_dir_name in enumerate(SCC):
    image_name = image_dir_name.split("\\")[-1]
    dataframe = dataframe.append([{"image_ID": image_name, "class": "SCC&AdC", "label": int(3)}], ignore_index=True)
    print(f"SCC&AdC： {i}/{len(SCC)}")


new_image_dir = os.path.join(new_dir, "image")
if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)
dataframe.to_excel(os.path.join(new_dir, "All_image_name_and_class.xlsx"))

All = ASC_H + ASC_U + NILM + SCC
for image_dir in All:
    shutil.copy(image_dir, new_image_dir)

