#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import pandas as pd
import numpy as np
from cross_validation import CrossValidation
import openai
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import sys
sys.path.insert(0, './models')
from funs import read_file



def main():
    """
    This script prepares the data for cross-validation for embeddings.

    It reads the input data, performs cross-validation, saves the train and test sets for each fold,
    """

    # Set up command line argument parser
    parser=argparse.ArgumentParser(description="Cross validation for embeddings",
                                   prog = "Cross Validator for embeddings",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds used in cross validation")
    parser.add_argument("--embedded_path", type=str, default='./data_preparation/output/datasets/COVID/COVID_pre_emb.npz',
                        help="path to save the embedded dataset")
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/COVID/cross_validation_datasets',
                        help="path to save the n folds cross-validation sets")

    args = parser.parse_args()
    NFOLD = args.num_fold
    SAVE_CROSS_DIR = os.path.join(args.save_dir, 'general')
    # Create the directory to save preprocessed cross-validation data if it doesn't already exist
    if not os.path.exists(SAVE_CROSS_DIR):
        os.makedirs(SAVE_CROSS_DIR)

    # Load the data and embeddings
    embedding_data = np.load(args.embedded_path)

    # Create an instance of the CrossValidation class and perform cross-validation
    cv = CrossValidation(n_splits=NFOLD, save_path=SAVE_CROSS_DIR)
    splits = cv.load_splits(save_split_file = 'cross_validation_splits.pkl')

    # Iterate through the splits and save the train and test sets for each fold
    for i, (train_index, test_index) in enumerate(splits):
        np.savez(os.path.join(SAVE_CROSS_DIR, f"embed_train_{i}.npz"),
                 embedding =  embedding_data["embedding"][train_index],
                 PMID = embedding_data["PMID"][train_index])

        np.savez(os.path.join(SAVE_CROSS_DIR, f"embed_test_{i}.npz"),
                 embedding =  embedding_data["embedding"][test_index],
                 PMID = embedding_data["PMID"][test_index])
    # Print a message indicating that preprocessing is complete
    print("Cross-Validation for embeddings completed !!!!!!!")
if __name__ == '__main__':
    main()





