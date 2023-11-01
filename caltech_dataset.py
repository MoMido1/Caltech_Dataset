from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        # super(Caltech,self) creates a super object that represents the parent class of Caltech and it is used to 
        # access methods and attributes from the parent class like in out case the init method
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.root = root
        file_path = os.path.join(root , split +'.txt')  # Adjust the path as needed

        # Check if the file exists before attempting to open it
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Read the contents of the file
                file_contents = file.read()
                file_contents = file_contents.split()
                data = np.empty((len(file_contents),2),dtype=object)
                i =0
                for line in file_contents:
                    parts = line.split('/')
                    data[i][0] = parts[0]
                    data[i][1] = parts[1]
                    i+=1
                self.data = data
                # return data
        else:
                print(f"The file '{file_path}' does not exist.")
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        pth = os.path.join(self.root,'101_ObjectCategories',self.data[index,0],self.data[index,1])
        # os.path.join(root , split +'.txt')
        image = pil_loader(pth)
        label = self.data[index,0]
        # image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        
        length =  self.data.shape[0]
        return length


# current_file_path = os.path.abspath(__file__)
# current_directory = os.path.dirname(current_file_path)

# train_dataset = Caltech(current_directory, split='train')

# img,lbl = train_dataset.__getitem__(1)
# print(img)
# print(lbl)
# len = train_dataset.__len__()
# print(len)
