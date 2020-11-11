#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0
import os 
import logging
import argparse
import jsonlines
import json
from itertools import chain
import numpy as np
import pandas as pd
from pandarallel import pandarallel

import nltk
import sklearn

from search_utils import helpers, search_preprocessing

nltk.download('punkt')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.2)
    parser.add_argument('--total-nb-of-records', type=int, default=1000)
    args, _ = parser.parse_known_args()
    #initialise pandarallel
    pandarallel.initialize(progress_bar=False, use_memory_fs=False)
    
    logger.info(f'Received arguments {args}')

    input_data_path = os.path.join('/opt/ml/processing/input', 'data.csv')
    
    logger.info("-------------------Reading and processing data----------------")

    
    logger.info(f'Reading input data from {input_data_path}')
    data = pd.read_csv(input_data_path, index_col=0)

    textual_columns = ["processed_title"]
    features_columns = ["id","category_id","text_information"]
    category_column = "product_category"
    
    MIN_NB_PRODUCTS_PER_CAT = 10
    data = search_preprocessing.pre_model_data_preprocessing(data, textual_columns, features_columns, category_column)
    
    logger.info("Spliting the data to training and test sets")

    pc_test = float(args.train_test_split_ratio)
    train_data, test_data, train_cat, test_cat = sklearn.model_selection.train_test_split(data[features_columns], data[["category_id"]],\
                                                                  test_size=pc_test, stratify=data[["category_id"]])

    logger.info("-------------------Generating positive and negative pairs----------------")

    limits = {"TOTAL_NB_OF_RECORDS" : int(args.total_nb_of_records), "PC_POSITIVE" : 0.5}

    train_sentences_data_negative, train_sentences_data_positive,\
    train_negative_indices, train_positive_indices =\
    search_preprocessing.generate_sentence_pairs(train_data, limits)

    logger.info(f"Number of negative data points : {len(train_sentences_data_negative)}\nNumber of positive data points : {len(train_sentences_data_positive)}")

    limits["TOTAL_NB_OF_RECORDS"] = pc_test*limits["TOTAL_NB_OF_RECORDS"]

    test_sentences_data_negative, test_sentences_data_positive, \
    test_negative_indices, test_positive_indices = search_preprocessing.generate_sentence_pairs(test_data, limits)

    logger.info(f"Number of negative data points : {len(test_sentences_data_negative)}\nNumber of positive data points : {len(test_sentences_data_positive)}")    

    #Combining positive and negative pairs and shuffling data
    training_records = train_sentences_data_negative + train_sentences_data_positive 
    np.random.shuffle(training_records)
    test_records = test_sentences_data_negative + test_sentences_data_positive
    np.random.shuffle(test_records)

    logger.info(f"Training records count : {len(training_records)}")
    logger.info(f"Test records count : {len(test_records)}")

    logger.info("Done preprocessing, performing data leak check...")
    data_leak_from_train_to_test = search_preprocessing.check_data_leak(training_records, test_records)
    logger.info(f"Data from train to test leaked ? {data_leak_from_train_to_test}")
    
    textual_train_data_output_path = os.path.join('/opt/ml/processing/train_textual', 'textual_train_data.jsonl')
    textual_test_data_output_path = os.path.join('/opt/ml/processing/test_textual', 'textual_test_data.jsonl')
    
    logger.info(f"Saving the textual training data to : {textual_train_data_output_path}")
    with jsonlines.open(textual_train_data_output_path, mode='w') as writer:
        for record in training_records:
            writer.write(record)
            
    logger.info(f"Saving the textual test data to : {textual_test_data_output_path}")
    with jsonlines.open(textual_test_data_output_path, mode='w') as writer:
        for record in test_records:
            writer.write(record)

    logger.info("-------------------Generating the vocabulary----------------")

    #The raw vocabulary, used for debuging and inspection
    raw_vocabulary_output_path = os.path.join("/opt/ml/processing/raw_vocab", "raw_vocab.json")
    
    #The final vocab, used for transforming text to ids
    vocabulary_output_path = os.path.join("/opt/ml/processing/vocab", "vocab.json")
        
    data_iter_list = []
    data_iter_list.append(helpers.read_jsonline(textual_train_data_output_path))
    data_iter_list.append(helpers.read_jsonline(textual_test_data_output_path))

    data_iter = chain(data_iter_list[0], data_iter_list[1])

    raw_vocab, word_to_id = search_preprocessing.build_vocab_parallel(data_iter,\
                   num_words=1000000, min_count=1, use_reserved_symbols=False, sort=True)

    logger.info("Generated vocabulary, saving vocabulary...")
    
    with open(vocabulary_output_path, "w") as write_file:
        json.dump(word_to_id, write_file)

    with open(raw_vocabulary_output_path, "w") as write_file:
        json.dump(raw_vocab, write_file)
    
    logger.info("-----------------Converting textual data to integers using generated vocabulary-------------------")

    #This tokenize will be used to process the textual data
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    numerical_train_data_output_path = os.path.join("/opt/ml/processing/train_numerical", "numerical_train_data.jsonl")
    numerical_test_data_output_path = os.path.join("/opt/ml/processing/test_numerical", "numerical_test_data.jsonl")

    logger.info("Converting training textual records to numerical records using the vocabulary")

    training_textual_records = pd.DataFrame(training_records)
    training_numerical_records = search_preprocessing.transform_textual_records_to_numerical(training_textual_records, tokenizer, word_to_id)

    with jsonlines.open(numerical_train_data_output_path, mode='w') as writer:
        for record in training_numerical_records:
            writer.write(record)

    logger.info("Converting test textual records to numerical records using the vocabulary")

    test_textual_records = pd.DataFrame(test_records)
    test_numerical_records = search_preprocessing.transform_textual_records_to_numerical(test_textual_records, tokenizer, word_to_id)

    with jsonlines.open(numerical_test_data_output_path, mode='w') as writer:
        for record in test_numerical_records:
            writer.write(record)

    logger.info("Converting to integers done")
    logger.info(f"Saving the numerical training data to : {numerical_train_data_output_path}")
    logger.info(f"Saving the numerical test data to :  {numerical_test_data_output_path}")
    logger.info("Done preprocessing")
