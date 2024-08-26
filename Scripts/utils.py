# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm

import pandas as pd
import numpy as np
import csv
import torch
import ctypes
import gc
import os

import os
from PIL import Image
from itertools import product
import matplotlib.pyplot as plt
import dataset
import torch
import warnings

import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
torch.manual_seed(42)
os.environ['PJRT_DEVICE'] = 'TPU' 
# os.environ['PT_XLA_DEBUG_LEVEL'] = '2'
# os.environ['XLA_DYNAMO_DEBU'] = '1'
# os.environ['PT_XLA_DEBUG'] = '1'
# os.environ['XLA_SYNC_WAIT'] = '1'
warnings.filterwarnings("ignore")
libc = ctypes.CDLL("libc.so.6")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block
import numpy as np
import torch

import pandas as pd
import numpy as np
import csv
import torch
import ctypes
import gc
import os

import torch
import warnings

import pandas as pd
import numpy as np
import torch.nn as nn
import timm
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torchvision
from torchview import draw_graph
import graphviz

import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import skimage.io as io
import os
import albumentations
import albumentations.pytorch
import random

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    # print(label.shape)
    # print(label_values)
    for colour in label_values:
        # print(colour)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values : values assignes to each position in one hot encoded vector

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
    
def to_tensor(x, **kwargs):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))
    
def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)

import random

class Custom_Dataset(Dataset):
    def __init__(self, img_dir, annotation_file, class_rgb_values = None, transforms = None, preprocessing = None):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.class_rgb_values = class_rgb_values
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = (np.array(io.imread(img_path)))
        
        annotation_ids = self.coco.getAnnIds(imgIds = img_info['id'])
        annotations = self.coco.loadAnns(annotation_ids)

        mask = np.zeros((img_info['height'], img_info['width']))

        random.shuffle(annotations)
        probability = random.randint(0, len(annotations))
        annotations = annotations[:probability]
        
        for annotation in annotations:
            # Get the segmentation polygons
            segmentation = annotation['segmentation']
            
            # If the segmentation is a list of polygons
            if isinstance(segmentation, list):
                rles = maskUtils.frPyObjects(segmentation, img_info['height'], img_info['width'])
                rle = maskUtils.merge(rles)
                mask += maskUtils.decode(rle)
            # If the segmentation is a RLE
            else:
                mask = maskUtils.merge([mask, maskUtils.decode(maskUtils.frPyObjects(segmentation, img_info['height'], img_info['width']))])

        mask = one_hot_encode(np.expand_dims(mask, axis = -1), self.class_rgb_values).astype('float')
        
        if self.transforms:
            transformed = self.transforms(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']
            
        mask[mask > 1] = 0
        mask = (mask).permute(2, 0, 1)
        sample = {'img' : image, "mask" : mask}
        
        return sample

def custom_loss(pred, target, metrics, bce_weight=0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    return bce * bce_weight + dice * (1 - bce_weight)

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return 1 - dice.mean()

def calculate_metrics(metrics, epoch_samples):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return outputs


def tile(filename, dir_in, dir_out, d):
    # Create the output directory if it doesn't exist
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    
    # Padding with 6 pixels around the image
    padding = 6
    padded_width = img.size[0] + 2 * padding
    padded_height = img.size[1] + 2 * padding
    
    # Create a new image with padding
    result = Image.new(img.mode, (padded_width, padded_height), (0, 0, 255)) 
    result.paste(img, (padding, padding))
    
    w, h = result.size
    fig, ax = plt.subplots(4, 4, figsize=(12, 6))
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}.png')
        tile = result.crop(box)
        tile.save(out)
        
        ax[i//128, j//128].imshow(tile)

def join_tiles_np(tiles, w, h, d):
    # Create an empty array with the dimensions of the original padded image
    result_array = np.zeros((h, w), dtype=tiles[0][2].dtype)
    
    # Iterate through the tiles and place them in the correct position
    for i, j, tile in tiles:
        result_array[i:i+d, j:j+d] = tile
    
    return result_array