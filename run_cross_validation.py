#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import pandas as pd
import numpy as np
from cross_validation import CrossValidation
from analyze_explanation import evaluate_explanations
import sys
sys.path.insert(0, './models')
from funs import setup_logger
from run_evaluation import save_mean_std_metrics
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)








def main():
    """
    This script prepares the data for cross-validation.
    It reads the input data, performs cross-validation, saves the train and test sets for each fold
    """

    # Set up command line argument parser
    parser=argparse.ArgumentParser(description="Cross validation",
                                   prog = "Cross Validator",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--num_fold", type=int, default=5, help="number of folds used in cross validation")
    parser.add_argument("--input_path",type=str, default='./data_preparation/output/datasets/COVID/COVID_pre_explanations.csv',
                        help = "path to saved dataset with generated explanations")
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/COVID/cross_validation_datasets',
                        help="path to save the n folds cross-validation sets")
    parser.add_argument("--log_path", type=str, default='./data_preparation/output/datasets/COVID/cross_validation_datasets/cross_validation.log', help="Logging file")

    args = parser.parse_args()

    # Create directories for storing the logs if they don't exist and set up a logger for logging progress and debugging info
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logger = setup_logger(args.log_path)

    # Define the directory where cross-validation splits will be saved, and create the cross-validation directory if it doesn't exist
    SAVE_CROSS_DIR = os.path.join(args.save_dir, 'general')
    if not os.path.exists(SAVE_CROSS_DIR):
        os.makedirs(SAVE_CROSS_DIR)

    # Load the dataset from the input path
    data = pd.read_csv(args.input_path)



    # Create an instance of the CrossValidation class and perform cross-validation
    cv = CrossValidation(n_splits=args.num_fold,save_train_test_split = True, save_path=SAVE_CROSS_DIR)
    # Generate the train/test splits for cross-validation
    splits = cv.split(df=data)

    # Define the explanation column names to be used in evaluation
    save_cols = ["Generated_justification", "Generated_zeroshot_cot", "Generated_zeroshot_cot_GPT4_A1","Generated_zeroshot_cot_GPT4_A2"]
    # Loop over each explanation column in save_cols to evaluate
    for save_col in save_cols:

        # Create empty lists to store metrics for each fold
        train_cv_metrics, test_cv_metrics = [], []

        # Iterate over each split
        for i, (train_index, test_index) in enumerate(splits):
            # Create training and testing dataframes based on the current split
            Train = data.iloc[train_index].copy()
            Test = data.iloc[test_index].copy()
            Train = Train[Train['Review_Paper'] == 'No'].copy()
            Test = Test[Test['Review_Paper'] == 'No'].copy()

            # Evaluate both the training and testing sets
            for dataset in ["train", "test"]:

                # Define the folder where results will be saved and create the results directory if it doesn't exist
                RESULT_FOLDER = os.path.join(args.save_dir, save_col, f'{dataset}_{i}')
                os.makedirs(RESULT_FOLDER, exist_ok=True)

                # Evaluate explanations and append the results to the appropriate list
                if dataset == "train":
                    metrics = evaluate_explanations(data=Train, save_col=save_col, RESULT_FOLDER=RESULT_FOLDER,
                                                    logger=logger)
                    train_cv_metrics.append(metrics)
                else:
                    metrics = evaluate_explanations(data=Test, save_col=save_col, RESULT_FOLDER=RESULT_FOLDER,
                                                    logger=logger)
                    test_cv_metrics.append(metrics)

        # Save the mean and standard deviation of the metrics across all folds for both training and testing sets for each explanation column.
        save_mean_std_metrics(cv_metrics=train_cv_metrics, saved_path=os.path.join(args.save_dir, save_col),
                              saved_csv=f"{save_col}_train_mean.csv")
        save_mean_std_metrics(cv_metrics=test_cv_metrics, saved_path=os.path.join(args.save_dir, save_col),
                              saved_csv=f"{save_col}_test_mean.csv")

    # Print a message indicating that preprocessing is complete
    print("Cross Validation Completed !!!!!!!")
if __name__ == '__main__':
    main()





