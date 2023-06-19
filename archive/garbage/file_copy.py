"""
앙상블시 오류방지
"""
import os
import shutil

folder1_path = r'C:\MB_Project\project\Competition\VISOL\tmp_t\0'
folder2_path = r"C:\MB_Project\project\Competition\VISOL\tmp_t\ensemble"
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

for file in folder1_files:
    if file not in folder2_files:
        file_path = os.path.join(folder1_path, file)
        shutil.copy2(file_path, folder2_path)