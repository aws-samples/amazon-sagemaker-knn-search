#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0
import os
import logging
import jsonlines
from itertools import chain, islice
from collections import Counter
import numpy as np
import pandas as pd
import multiprocessing


import sklearn
import nltk

nltk.download('punkt')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# constants
BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]

LABEL_DICT = {1:'positive', 0:'negative'}


def pre_model_data_preprocessing(data, textual_columns, features_columns, category_column):
    """
    This function is used ensure the same robust data processing and cleaning for both training and inference.
    It performs the following steps: 
    -Creating a common text_information column based on a list of defined columns (textual_columns)
    -Keeps only records where textual information is not missing
    -Drops duplicates and encodes the category values usign the sklearn label encoder

    Arguments:
        data {pandas.DataFrame} -- The input dataset
        textual_columns {List} -- The list of columns that contains textual information and will be aggregated into one column
        features_columns {List} -- The list of columns that will be used in the rest of the process
        category_column {List} -- The column used as a category

    Returns:
        pd.DataFrame -- The processed dataset containing additional textual column  "textual information"
    """

    logger.info(f"Shape of data given in input :{data.shape}")

    for column in textual_columns:
        data[column] = data[column].fillna("")
        data[column] = data[column].astype("str")
    
    
    data["text_information"] = data[textual_columns].agg(' '.join, axis=1)

    #Making sure there is no null values after processing..
    data = data[~data["text_information"].isnull()]
    data = data[data["text_information"]!=""]
    logger.info(f"After droping missing textual information: {data.shape}")

    
    #Droping duplicates 
    data = data.drop_duplicates(subset=['text_information'])
    logger.info(f"Total number of records after droping duplicates : {data.shape[0]}")
    
    
    #Keeping only the feature columns
    l_encoder = sklearn.preprocessing.LabelEncoder()
    data["category_id"] = l_encoder.fit_transform(data[category_column])
    
    data = data[features_columns]

    #Making sure feature columns are in string format
    for column in features_columns:
        data[column] = data[column].astype("str")
        
    data = data.reset_index(drop=True)

    return data

def get_tokens(line, tokenizer):
    """    
    Yields tokens from input string.

    Args:
        line {String} -- The input string
        tokenizer {nltk.tokenize.TreebankWordTokenizer} -- The tokenizer object allows to split a text string to a list of tokens

    Yields:
        token {String} -- Iterats over tokens
    """

    for token in tokenizer.tokenize(line):
        if len(token) > 0:
            yield token

def sentence_to_integers(sentence, tokenizer, word_dict):
    """
    Converts a sentence of text to a list of tokens using the word dictionary

    Arguments:
        sentence {string} -- [description]
        tokenizer {nltk.tokenize.TreebankWordTokenizer} -- The tokenizer object allows to split a text string to a list of tokens
        word_dict {dict} -- The vocabulary dictionary, this dictionary maps a word to it's id

    Returns:
        list -- a list of tokens
    """

    return [word_dict[token] for token in get_tokens(sentence, tokenizer)
            if token in word_dict]

def get_tokens_from_pairs(input_dict, tokenizer):
    """
    Parse the dictionary and tokenize the textual information inside

    Arguments:
        input_dict {dict} -- [description]
        tokenizer {nltk.tokenize.TreebankWordTokenizer} -- The tokenizer object allows to split a text string to a list of tokens

    Returns:
        Iterator -- An iterator containing all textual information
    """

    iter_list = []
    for sentence_key in ['left_text_information', 'right_text_information']:
        sentence = input_dict[sentence_key]
        iter_list.append(get_tokens(sentence, tokenizer))

    return chain(iter_list[0], iter_list[1])

def build_vocab(data):
    """
    Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
    using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
    (PAD).

    Arguments:
        data {list} : List of pairs 

    Returns:
        List -- a list of tokens
    """

    results = []
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    
    for line in data:
        for token in get_tokens_from_pairs(line, tokenizer):
            if token not in set(VOCAB_SYMBOLS):
                results.append(token)

    return results

def build_vocab_parallel(data_iter, num_words=50000, min_count=1, use_reserved_symbols=True,
                         sort=True):
    """
    Calls the "build_vocab()" in a parallel mode, splits the data_iter, after loading it; given as input
    based on the number of cpus available and calls the function

    Arguments:
        data_iter {Iterator} --  Sequence of sentences containing whitespace delimited tokens.

    Keyword Arguments:
        num_words {int} -- Maximum number of words in the vocabulary. (default: {50000})
        min_count {int} -- Minimum occurrences of words to be included in the vocabulary. (default: {1})
        use_reserved_symbols {bool} -- If we're using the reserver symbols dictionary (default: {True})
        sort {bool} -- If the vocab should be sorted or not (default: {True})

    Returns:
        dict --  The vocabulary dictionary, this dictionary maps a word to it's id

    """

    logger.info("Preparing data for prarallel processing")
    unloaded_lines = [e for e in data_iter]

    nb_cpus = multiprocessing.cpu_count()
    chunk_size = int(len(unloaded_lines) // nb_cpus)
    nb_chunks = int(len(unloaded_lines) / chunk_size)

    # This might fill up the memory, use carefully, check with htop
    list_datasets = []
    for i in range(nb_chunks + 1):
        sub_unloaded_lines = unloaded_lines[i * chunk_size: (i + 1) * chunk_size]
        list_datasets.append(sub_unloaded_lines)
        del sub_unloaded_lines
        
    logger.info("Launching parallel generation of vocab")
    logger.info(f"Number of cpus uses : {nb_cpus}")
    logger.info(f"Number of params combination : {len(list_datasets)}")
    
    p = multiprocessing.Pool(nb_cpus)
    results = p.map(build_vocab, list_datasets)
    p.close()

    all_tokens = []
    for result in results:
        all_tokens.extend(result)

    raw_vocab = Counter(all_tokens)

    logger.info(f"Initial vocabulary: {len(raw_vocab)} types")

    #For words with the same count, they will be ordered reverse alphabetically.
    pruned_vocab = sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)
    logger.info(f"Pruned vocabulary: {len(pruned_vocab)} types (min frequency {min_count})")

    #Truncate the vocabulary to fit size num_words (only includes the most frequent ones)
    vocab = islice((w for c, w in pruned_vocab), num_words)

    if sort:
        #Sort the vocabulary alphabetically
        vocab = sorted(vocab)
    if use_reserved_symbols:
        vocab = chain(set(VOCAB_SYMBOLS), vocab)

    word_to_id = {word: idx for idx, word in enumerate(vocab)}

    logger.info(f"Final vocabulary: {len(word_to_id)}")

    if use_reserved_symbols:
        # Important: pad symbol becomes index 0
        assert word_to_id[PAD_SYMBOL] == PAD_ID
    
    return raw_vocab, word_to_id

def convert_text_to_integers(data_iter, word_to_id, output_path):
    """
    This function converts the left and right pairs from the textual information
    to numerical values using the vocabulary provided

    Args:
        data_iter {Iterator} --  Sequence of sentences containing whitespace delimited tokens.
        word_to_id {dict} -- The vocabulary dictionary, this dictionary maps a word to it's id
        output_path {String} -- The path to which the outputs are saved
    """

    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    count = 0
    max_seq_length = 0
    with jsonlines.open(output_path, mode='w') as writer:
        for in_dict in data_iter:
            out_dict = dict()
            label = in_dict['pair_label']  
            if label in LABEL_DICT:
                rsentence1 = in_dict['left_text_information']
                rsentence2 = in_dict['right_text_information']
                for idx, sentence in enumerate([rsentence1, rsentence2]):
                    # logger.info(count, sentence)
                    s = sentence_to_integers(sentence, tokenizer, word_to_id)
                    out_dict[f'in{idx}'] = s
                    max_seq_length = max(len(s), max_seq_length)
                out_dict['pair_label'] = label  
                writer.write(out_dict)
            else:
                logger.info(label)
                count += 1

    logger.info(f"There are in total {count} invalid labels")
    logger.info(f"The max length of converted sequence is {max_seq_length}")

    return 

def generate_negative_pairs_for_cat(X, category_id, max_negative_per_cat, nb_categories):
    """[summary]

    Args:
        X {pd.Dataframe}: The dataframe from which we'll generate the pairs
        category_id {String}: The category id for which we want to generate negative pairs
        max_negative_per_cat {int}: The maximum number of samples we'll generate for this category
        nb_categories {int}: The total number of categories in the dataset

    Returns:
        pd.Dataframe: A pandas dataframe containing left and right product information (i.e: id, textual information, category)
    """

    logger.info(f"Generating negative pairs for category nb : {category_id}")
    X_same_cat = X[X["category_id"]==category_id]
    X_diff_cat = X[X["category_id"]!=category_id]

    if len(X_same_cat) < max_negative_per_cat:
        sample_left_pair = X_same_cat.sample(max_negative_per_cat, replace=True)
    else:
        sample_left_pair = X_same_cat.sample(max_negative_per_cat)

    if len(X_diff_cat) < max_negative_per_cat:
        sample_right_pair = X_diff_cat.sample(max_negative_per_cat, replace=True)
    else:
        sample_right_pair = X_diff_cat.sample(max_negative_per_cat) 

    sample_left_pair = sample_left_pair.rename(columns={"id":"left_id","category_id":"left_category_id",\
        "text_information":"left_text_information"})
    sample_right_pair = sample_right_pair.rename(columns={"id":"right_id","category_id":"right_category_id",\
        "text_information":"right_text_information"})

    sample_left_pair = sample_left_pair.reset_index(drop=True)
    sample_right_pair = sample_right_pair.reset_index(drop=True)
    product_pairs = pd.concat([sample_left_pair, sample_right_pair], axis=1)
    
    return product_pairs

def generate_sentence_pairs(X, limits):
    """
    This function creates negative and positive pairs, it uses a stratified sampling to ensure
    representation of categories distribution in the data.
    Instead of creating all possible pairs then scoping down, we take the opposit approach working backwards
    from the target number of records we aim to generate


    Arguments:
        X {pd.DataFrame} -- The data scoped down to "feature_columns"
        limits {dict} --  A dictionary of limits containing "TOTAL_NB_OF_RECORDS" and "PC_POSITIVE" where:
                        - TOTAL_NB_OF_RECORDS : The total number of positive and negative pairs we want to generate
                        - PC_POSITIVE : The percentage of the positive pairs from TOTAL_NB_OF_RECORDS
        

    Returns:
         sentences_data_negative : The list of pairs that represent a negative combination (see note)
         sentences_data_positive : The list of pairs that represent a positive combination (see note)
         negative_indices : The list of unique ids that represent negative indices 
         positive_indices : The list of unique ids that represent positive indices 

    Note - Each pair contains the following :contains the following keys: "pair_id", "pair_label", 
    "left_id", "left_category_id", "left_text_information", "right_id", "right_category_id", "right_text_information"
    """

    logger.info("Using the following limits to generate pairs :")
    logger.info(limits)
    logger.info("----------------------------------------------")
    
    TOTAL_NB_OF_RECORDS = limits["TOTAL_NB_OF_RECORDS"]
    PC_POSITIVE = limits["PC_POSITIVE"]
    TOTAL_NB_OF_POSITIVE_RECORDS = int(np.ceil(PC_POSITIVE*TOTAL_NB_OF_RECORDS))
    TOTAL_NB_OF_NEGATIVE_RECORDS = TOTAL_NB_OF_RECORDS - TOTAL_NB_OF_POSITIVE_RECORDS

    category_ids = X["category_id"].unique()
    nb_categories = len(category_ids)

    max_positive_per_cat = int(np.ceil(TOTAL_NB_OF_POSITIVE_RECORDS/nb_categories))
    max_negative_per_cat = int(np.ceil(TOTAL_NB_OF_NEGATIVE_RECORDS/nb_categories))

    logger.info(f"Number of categories is : {nb_categories}")
    logger.info(f"Number of requested records is : {TOTAL_NB_OF_RECORDS}")
    logger.info(f"Number of positive pairs to generate per category are  : {max_positive_per_cat}")
    logger.info(f"Number of negative pairs to generate per category are  : {max_negative_per_cat}")

    if max_negative_per_cat != max(max_negative_per_cat, nb_categories):
        logger.info('Warning, the max negative pair per cat is lower than the total number of categories')  
        
    #Generating positive pair for each category id
    positive_product_pairs = []
    for category_id in X["category_id"].unique():
        logger.info(f"Generating positive pairs for category nb : {category_id}")
        X_same_cat = X[X["category_id"]==category_id]
        if len(X_same_cat) < max_positive_per_cat:
            sample_left_pair = X_same_cat.sample(max_positive_per_cat, replace=True)
            sample_right_pair = X_same_cat.sample(max_positive_per_cat, replace=True)
        else:
            sample_left_pair = X_same_cat.sample(max_positive_per_cat)
            sample_right_pair = X_same_cat.sample(max_positive_per_cat)        

        sample_left_pair = sample_left_pair.rename(columns={"id":"left_id","category_id":"left_category_id",\
            "text_information":"left_text_information"})
        sample_right_pair = sample_right_pair.rename(columns={"id":"right_id","category_id":"right_category_id",\
            "text_information":"right_text_information"})

        sample_left_pair = sample_left_pair.reset_index(drop=True)
        sample_right_pair = sample_right_pair.reset_index(drop=True)
        product_pairs = pd.concat([sample_left_pair, sample_right_pair],axis=1)
        positive_product_pairs.append(product_pairs)
    
    #Concatenating and adding unique pair ids and label=1
    positive_product_pairs = pd.concat(positive_product_pairs)
    positive_product_pairs = positive_product_pairs.reset_index(drop=True)
    positive_product_pairs["pair_id"] = range(len(positive_product_pairs))
    positive_product_pairs["pair_label"] = 1
    
    #Generating negative pair for each category id
    negative_product_pairs = []
    for category_id in category_ids:
        negative_product_pairs.append(generate_negative_pairs_for_cat(X, category_id, max_negative_per_cat, nb_categories))

    #Concatenating and adding unique pair ids and label=0
    negative_product_pairs = pd.concat(negative_product_pairs, axis=0)
    negative_product_pairs = negative_product_pairs.reset_index(drop=True)
    negative_product_pairs["pair_id"] = range(len(negative_product_pairs))
    negative_product_pairs["pair_label"] = 0
    
    logger.info("Performing checks on generated pairs...")
    
    if np.sum(positive_product_pairs["left_category_id"]==positive_product_pairs["right_category_id"]) == positive_product_pairs.shape[0]:
        logger.info("Check passed : All positive pairs have the same category ids")
    else:
        #Raising an exception would be better
        logger.warning("Check failed : Warning, not all positive pairs have the same category ids")

    if np.sum(negative_product_pairs["left_category_id"]==negative_product_pairs["right_category_id"]) == 0:
        logger.info("Check passed : All negative pairs have different category ids")
    else:
        #Raising an exception would be better
        logger.warning("Check failed : Warning, not all negative pairs have different category ids")

    #Transforming to json records 
    sentences_data_positive = positive_product_pairs.to_dict('records')
    sentences_data_negative = negative_product_pairs.to_dict('records')
    positive_indices = positive_product_pairs["pair_id"].values
    negative_indices = negative_product_pairs["pair_id"].values

    return sentences_data_negative, sentences_data_positive, negative_indices, positive_indices

def transform_textual_records_to_numerical(textual_records, tokenizer, word_to_id):
    """
    This functions transforms the textual infromation in each record to numerical
    information using the word_to_id vocabulary and the tokenizer passed in parameters

    Arguments:
        textual_records {List} -- A list of pairs where each pair contains two textual descriptions and a label 
        tokenizer {nltk.tokenize.TreebankWordTokenizer} -- The tokenizer object allows to split a text string to a list of tokens
        word_to_id {dict} -- The vocabulary dictionary, this dictionary maps a word to it's id

    Returns:
        A List of pairs where each textual pair is replaced by a numerical pair 
    """

    textual_records["in0"] = textual_records["left_text_information"].parallel_apply(lambda x:sentence_to_integers(x, tokenizer, word_to_id))
    textual_records["in1"] = textual_records["right_text_information"].parallel_apply(lambda x:sentence_to_integers(x, tokenizer, word_to_id))

    numerical_records = textual_records[["in0","in1","pair_label"]]
    numerical_records = numerical_records.rename(columns={"pair_label":"label"})

    numerical_records = numerical_records.to_dict('records')
    
    return numerical_records

def check_data_leak(training_records, test_records):
    """
    This function is used to ensure there is no records being present in both the training and test records

    Arguments:
        training_records {List} -- The list of training pairs (see note)
        test_records {List} -- The list of test pairs (see note)

    Returns:
        boolean -- True meaning that there a pair that is present both in trainign and test, False otherwise.

    Note - Each pair contains the following :contains the following keys: "pair_id", "pair_label", 
    "left_id", "left_category_id", "left_text_information", "right_id", "right_category_id", "right_text_information"
    """

    train_sentence_ids = []
    for e in training_records:
        train_sentence_ids.append(e["left_id"])
        train_sentence_ids.append(e["right_id"])

    train_sentence_ids = np.unique(train_sentence_ids)

    test_sentence_ids = []
    for e in test_records:
        test_sentence_ids.append(e["left_id"])
        test_sentence_ids.append(e["right_id"])

    test_sentence_ids = np.unique(test_sentence_ids)

    if len(set(train_sentence_ids).intersection(test_sentence_ids)) == 0:
        return False
    else:
        return True