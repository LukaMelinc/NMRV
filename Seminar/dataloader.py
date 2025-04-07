import cv2
from torchvision.transforms import transforms
import json
from torch.utils.data import Dataset, DataLoader


class CustomDataloader(Dataset):
    def __init__(self, json_file, transform=None, images=None):
        self.data = self.load_dataset(json_file)
        self.transform = transform
        self.images = images

    def load_data(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        image = cv2.imread(image_path)

        if 'thermal' in item:
            image = preprocess_thermal(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = item['annotations']

        return image, annotation
    



def preprocess_thermal(thermal):

    thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2RGB)
    
    thermal = cv2.normalize(thermal, dst=None, alpha=0, beta=65535,
    
    norm_type=cv2.NORM_MINMAX)
    
    thermal = cv2.convertScaleAbs(thermal, alpha=255/(2**16))
    
    return thermal

def create_dataloader(json_file, batch_size=4, shuffle=True, num_workers=2):
    dataset = CustomDataloader(json_file, transform=None)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def load_split_files(train_file, test_file):
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(test_file, 'r') as g:
        test_data = json.load(g)


    return train_data, test_data


    