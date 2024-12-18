import torch
import torch.utils.data as data
import pickle
import os
import numpy as np
from skimage.color import rgb2lab
import warnings
import open_clip
from tqdm import tqdm
from PIL import Image

class PAT_Dataset(data.Dataset):
    def __init__(self, src_path, trg_path, input_dict):
        with open(src_path, 'rb') as fin:
            self.src_seqs = pickle.load(fin)
        with open(trg_path, 'rb') as fin:
            self.trg_seqs = pickle.load(fin)

        words_index = []
        for index, palette_name in enumerate(self.src_seqs):
            temp = [0] * input_dict.max_len

            for i, word in enumerate(palette_name):
                temp[i] = input_dict.word2index[word]
            words_index.append(temp)
        self.src_seqs = torch.LongTensor(words_index)

        palette_list = []
        for index, palettes in enumerate(self.trg_seqs):
            temp = []
            for palette in palettes:
                rgb = np.array([palette[0], palette[1], palette[2]]) / 255.0
                warnings.filterwarnings("ignore")
                lab = rgb2lab(rgb[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
                temp.append(lab[0])
                temp.append(lab[1])
                temp.append(lab[2])
            palette_list.append(temp)

        self.trg_seqs = torch.FloatTensor(palette_list)
        self.num_total_seqs = len(self.src_seqs)

    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs

class Text2PaletteDataset(data.Dataset):
    """
    Data loader for the text and RGB palette dataset.
    Each point involves a text description and a corresponding RGB palette, defined as 
    as 5 3-dimensional vectors (5 colors in RGB format)
    """
    MODEL_NAME = 'ViT-B-32'

    def __init__(self, embedding_file, palette_file, batch_size):
        
        # Load precomputed embeddings as a PyTorch tensor
        self.text_embeddings = torch.load(embedding_file)  # Shape: [num_samples, embedding_dim]
        
        with open(palette_file, 'rb') as f:
            self.palette_data = np.asarray(pickle.load(f)).reshape(-1, 5, 3) / 255.0

        self.palette_data = torch.Tensor(self.palette_data).to(torch.float32)

        self.data_size = len(self.text_embeddings)
        self.batch_size = batch_size
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # Retrieve precomputed CLIP embedding for the text prompt
        text_emb = self.text_embeddings[idx]
        
        # Retrieve RGB palette
        palette = self.palette_data[idx]
        
        return text_emb, palette

    def to_data_loader(self):
        # Return DataLoader object with this dataset
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=True)
    
class ColorizationDataset(data.Dataset):
    """
    Main dataset class to train the GAN model for colorization
    """
    # TODO: Implement this class

    def __init__(self, image_path, palette_path, batch_size, num_workers,
                 cap:int = None):
        self.img_path = image_path
        self.palette_path = palette_path

        # load filenames for images and palettes
        self.image_filenames = os.listdir(image_path)
        self.palette_filenames = os.listdir(palette_path)

        if cap:
            self.image_filenames = self.image_filenames[:cap]
            self.palette_filenames = self.palette_filenames[:cap]

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # load image as a torch tensor
        img = Image.open(os.path.join(self.img_path, self.image_filenames[idx]))
        img = np.array(img)[:, :, :3] / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img).to(torch.float32)

        # load palette as a torch tensor
        palette = np.load(os.path.join(self.palette_path, self.palette_filenames[idx]))
        palette = rgb2lab(np.asarray(palette)
                                    .reshape(-1, 5, 3) / 256
                                    , illuminant='D50')
        
        palette = torch.Tensor(palette).to(torch.float32)

        return img, palette

    def to_data_loader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
class Image_Dataset(data.Dataset):
    def __init__(self, image_dir, pal_dir, cap:int = 1000):
        with open(image_dir, 'rb') as f:
            _l = pickle.load(f)
            if cap:
                _l = _l[:cap]
            self.image_data = np.asarray(_l) / 255

        with open(pal_dir, 'rb') as f:
            _l = pickle.load(f)
            if cap:
                _l = _l[:cap]

            self.pal_data = rgb2lab(np.asarray(_l)
                                    .reshape(-1, 5, 3) / 256
                                    , illuminant='D50')

        self.data_size = self.image_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.image_data[idx], self.pal_data[idx]
    
def p2c_loader(batch_size, cap:int = None):
    train_img_path = './data/bird256/train_palette/images'
    train_pal_path = './data/bird256/train_palette/palettes'

    train_dataset = ColorizationDataset(train_img_path, train_pal_path, cap=cap,
                                        batch_size=batch_size, num_workers=4)\
    
    train_loader = train_dataset.to_data_loader()

    imsize = 256
    return train_loader, imsize