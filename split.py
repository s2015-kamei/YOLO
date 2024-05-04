import os
import random
import shutil

def split_data(data_dir, train_ratio):
    # Create directories for training and testing data
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all files in the data directory
    files = os.listdir(data_dir)

    # Shuffle the files randomly
    random.shuffle(files)

    # txtファイルのみを抽出
    files = [f for f in files if f.endswith('.txt')]

    
    
    # Calculate the number of files for training and testing
    num_train = int(len(files) * train_ratio)
    num_test = len(files) - num_train

    print(len(files), num_train)

    # Move files to the training directory
    for file in files[:num_train]:
        src = os.path.join(data_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)

        png_src = os.path.join(data_dir, os.path.splitext(file)[0] + '.png')
        png_dst = os.path.join(train_dir, os.path.splitext(file)[0] + '.png')
        shutil.copy(png_src, png_dst)


    # Move files to the testing directory
    for file in files[num_train:]:
        src = os.path.join(data_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy(src, dst)

        png_src = os.path.join(data_dir, os.path.splitext(file)[0] + '.png')
        png_dst = os.path.join(test_dir, os.path.splitext(file)[0] + '.png')
        shutil.copy(png_src, png_dst)

# Usage example
data_dir = '/home/roboken03/Downloads/Data/'  # Replace with the path to your data directory
train_ratio = 0.8  # 80% of the data will be used for training

split_data(data_dir, train_ratio)