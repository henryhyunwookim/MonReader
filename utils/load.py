import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def load_data(file_name, folder_name=None):
    parent_path = Path(os.getcwd()).parent
    if folder_name:
        # path = parent_path / folder_name / file_name
        path = Path("/".join([os.getcwd(), folder_name, file_name]))
    else:
        # path = parent_path / file_name
        path = Path("/".join([os.getcwd(), file_name]))
    
    file_type = file_name.split(".")[1]
    if file_type=="csv":
        output = pd.read_csv(path)
    elif file_type== "xlsx":
        output = pd.read_excel(path)
    else:
        raise f"Failed to load data - invalid file type {file_type}"
    
    print(output.info(), "\n")
    print(output.describe(), "\n")

    return output


def check_file_downloaded(file_name, default_path, download_path):
    os.chdir(download_path)
    if os.path.exists(file_name):
        print(f"File {file_name} exist in {download_path}!")
    else:
        print(f"File {file_name} doesn't exist in {download_path}!")

    os.chdir(default_path)


def extract_zip_file(file_path, download_path, file_name):
    if os.path.exists(download_path + "\\" + file_name.split(".")[0]):
        print(f"{file_name} already extracted in {download_path}.")
    else:
        with ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(path=download_path)
        print(f"{file_name} extracted in {download_path}.")