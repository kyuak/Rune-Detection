import os
import glob

def count_txt_files(folder_path):
    png_files = glob.glob(os.path.join(folder_path, '*.txt'))
    count = len(png_files)
    return count

def count_jpg_files(folder_path):
    png_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    count = len(png_files)
    return count

folder_path1 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/images/train'
folder_path2 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/images/val'
folder_path3 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/images/test'

folder_path4 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/labels/train'
folder_path5 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/labels/val'
folder_path6 = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/labels/test'

count1 = count_jpg_files(folder_path1)
count2 = count_jpg_files(folder_path2)
count3 = count_jpg_files(folder_path3)
count4 = count_txt_files(folder_path4)
count5 = count_txt_files(folder_path5)
count6 = count_txt_files(folder_path6)

print(f"Number of files in {folder_path1}: {count1}")
print(f"Number of files in {folder_path2}: {count2}")
print(f"Number of files in {folder_path3}: {count3}")
print(f"Number of files in {folder_path4}: {count4}")
print(f"Number of files in {folder_path5}: {count5}")
print(f"Number of files in {folder_path6}: {count6}")