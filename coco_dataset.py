################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
from vqaTools.vqa import VQA
import string

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, annotation, ques_ids, vocab, img_size, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformations.
        """
        self.root = root
        self.vqa = VQA(annotation[0], annotation[1])
        self.ques_ids = ques_ids
        self.vocab = vocab
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if transform is not None:          
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p = 0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

        self.resize = transforms.Compose(
            [transforms.Resize(img_size, interpolation=2), transforms.CenterCrop(img_size)])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vqa = self.vqa
        vocab = self.vocab
        
        ques_id = self.ques_ids[index]
        question = vqa.qqa[ques_id]['question']
        answers = vqa.qa[ques_id]['answers']
        answer = answers[np.random.randint(len(answers))]['answer']
        
        image_id = vqa.qqa[ques_id]['image_id']
        path = self.root + str(image_id).zfill(12) + '.jpg'
        image = Image.open(path).convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))

        # Convert caption (string) to word ids.
        question = str(question).lower()
        question = question.translate(str.maketrans('', '', string.punctuation))
        
        tokens = nltk.tokenize.word_tokenize(question)
        question = [vocab('<start>')]
        question.extend([vocab(token) for token in tokens])
        question.append(vocab('<end>'))
        target = torch.Tensor(question)
        return image, target, ques_id

    def __len__(self):
        return len(self.ques_ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    by padding the captions to make them of equal length.

    We can not use default collate_fn because variable length tensors can't be stacked vertically.
    We need to pad the captions to make them of equal length so that they can be stacked for creating a mini-batch.

    Read this for more information - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_ids = zip(*data)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, img_ids
