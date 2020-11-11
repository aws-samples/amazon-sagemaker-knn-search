#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0
import logging
import os
import argparse
import json

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

nltk.download('punkt')
nltk.download('wordnet')

def trim_w2v(in_path, out_path, word_dict):
    """
    Arguments:
        in_path {string} -- The local path in which the glove dict is saved
        out_path {string} -- The local path in which the trimmed vocabulary will be written
        word_dict {dict} -- The vocabulary dictionary, this dictionary maps a word to it's id

    # credit: This preprocessing function is modified from the w2v preprocessing script in Facebook infersent codebase
    # Infersent code license can be found at: https://github.com/facebookresearch/InferSent/blob/master/LICENSE
    """

    lines = []
    with open(out_path, 'w') as outfile:
        with open(in_path) as f:
            for line in f:
                word, _ = line.split(' ', 1)
                if word in word_dict:
                    lines.append(line)

        logger.info('Found %s(/%s) words with w2v vectors' % (len(lines), len(word_dict)))
        outfile.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()

    logger.info(f'Received arguments {args}')

    input_glove_path = os.path.join('/opt/ml/processing/input_glove', 'glove.840B.300d.txt')
    input_vocabulary_path = os.path.join('/opt/ml/processing/input_vocabulary', 'vocab.json')

    logger.info(f'Reading glove artefact from {input_glove_path}')
    logger.info(f'Reading vocabulary data from {input_vocabulary_path}')
    with open(input_vocabulary_path) as f:
        word_dict = json.load(f)

    trimmed_glove_output_path = os.path.join('/opt/ml/processing/trimmed_glove', 'trimmed_glove.txt')

    logger.info(f'Trimming the glove embeddings and saving them to : {trimmed_glove_output_path}')
    trim_w2v(input_glove_path, trimmed_glove_output_path, word_dict)

    logger.info("Saving the vocab given as input")
    vocabulary_output_path = os.path.join('/opt/ml/processing/vocab', 'vocab.json')

    with open(vocabulary_output_path, "w") as write_file:
        json.dump(word_dict, write_file)

    logger.info("Done generating and saving trimmed glove embeddings")