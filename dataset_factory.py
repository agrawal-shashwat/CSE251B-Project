################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import csv, os
from torch.utils.data import DataLoader
from vqaTools.vqa import VQA

from vocab import load_vocab
from coco_dataset import CocoDataset, collate_fn


# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    images_root_dir = config_data['dataset']['images_root_dir']
    root_train = images_root_dir + 'train2014/COCO_train2014_'
    root_val = images_root_dir + 'train2014/COCO_train2014_'
    root_test = images_root_dir + 'val2014/COCO_val2014_'

    train_annotation_file = 'annotations/mscoco_train2014_annotations.json'
    train_question_file = 'annotations/MultipleChoice_mscoco_train2014_questions.json'
    test_annotation_file = 'annotations/mscoco_val2014_annotations.json'
    test_question_file = 'annotations/MultipleChoice_mscoco_val2014_questions.json'
    
    vqa_test = VQA(test_annotation_file, test_question_file)
    vqa_train = VQA(train_annotation_file, train_question_file)
    
    ques = vqa_train.getQuesIds(ansTypes = config_data['dataset']['ques_type'])
    ques_test = vqa_test.getQuesIds(ansTypes = config_data['dataset']['ques_type'])
    ques_train = ques[:int(0.8 * len(ques))]
    ques_val = ques[int(0.8 * len(ques)):]

    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    vocabulary = load_vocab(vqa_train, ques_train, vocab_threshold)

    train_data_loader = get_coco_dataloader(root_train, (train_annotation_file, train_question_file),
                                            vocabulary, config_data, ques_train, True)
    val_data_loader = get_coco_dataloader(root_val, (train_annotation_file, train_question_file),
                                          vocabulary, config_data, ques_val)
    test_data_loader = get_coco_dataloader(root_test, (test_annotation_file, test_question_file),
                                           vocabulary, config_data, ques_test)

    return vqa_test, vocabulary, train_data_loader, val_data_loader, test_data_loader


def get_coco_dataloader(imgs_root_dir, annotation, vocabulary, config_data, ques, trans = None):
    dataset = CocoDataset(root = imgs_root_dir,
                          annotation = annotation,
                          ques_ids = ques,
                          vocab = vocabulary,
                          img_size = config_data['dataset']['img_size'],
                         transform = trans)
    return DataLoader(dataset=dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config_data['dataset']['num_workers'],
                      collate_fn=collate_fn,
                      pin_memory=True)
