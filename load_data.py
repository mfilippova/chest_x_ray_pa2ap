import pydicom
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import zipfile


def unzip_data():
    with zipfile.ZipFile('stage_2_train_images.zip', 'r') as zip_ref:
        zip_ref.extractall('dicom_images')


def create_target_file():
    data = []
    for image_path in tqdm(glob.glob('dicom_images/*.dcm')):
        image = pydicom.read_file(image_path)
        view_position = image['ViewPosition'].value
        image_name = image_path.split('/')[-1].split('.')[0]
        data.append((image_name, view_position))
    
    data = pd.DataFrame(data, columns=['name', 'position'])
    data.to_csv('position.csv', index=False)


def main():
    unzip_data()
    create_target_file()


if __name__ == '__main__':
    main()
