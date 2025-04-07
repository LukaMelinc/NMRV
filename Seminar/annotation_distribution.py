import os
import shutil

def distribute_annotations(train_txt, test_txt, annotations_dir, train_dir, test_dir):
    # Read the train and test file lists
    with open(train_txt, 'r') as f:
        train_files = f.read().splitlines()

    with open(test_txt, 'r') as f:
        test_files = f.read().splitlines()

    # Ensure the train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Distribute the annotation files for training
    for file_name in train_files:
        src_file = os.path.join(annotations_dir, file_name + '.txt')
        dest_file = os.path.join(train_dir, file_name + '.txt')
        if os.path.exists(src_file):
            shutil.move(src_file, dest_file)
        else:
            print(f"Warning: {src_file} does not exist and will be skipped.")

    # Distribute the annotation files for testing
    for file_name in test_files:
        src_file = os.path.join(annotations_dir, file_name + '.txt')
        dest_file = os.path.join(test_dir, file_name + '.txt')
        if os.path.exists(src_file):
            shutil.move(src_file, dest_file)
        else:
            print(f"Warning: {src_file} does not exist and will be skipped.")

# Example usage
train_txt_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/train.txt'
test_txt_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/test.txt'
annotations_dir_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/annotations'
train_dir_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/thermal_images/train/train_annotations'
test_dir_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/thermal_images/test/test_annotations'

distribute_annotations(train_txt_path, test_txt_path, annotations_dir_path, train_dir_path, test_dir_path)
