#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime
import argparse
import os
import sys
import json
sys.path.insert(0, './models')
from run_evaluation import get_ans_percentage, ModelEvaluator
from funs import setup_logger
from parse_subquestion_json import JsonProcessor

def parse_args():
    parser=argparse.ArgumentParser(description="Explanation Analysis",
                               prog = "Explanation Analysis",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input_dir", type=str, default='./data_preparation/output/datasets/Q2_wetlab/Nipah', help="Path to the input csv files")
    parser.add_argument("--explanation_dict", type=str, default="./data_preparation/output/datasets/Q2_wetlab/Nipah/explanation_dict.json", help="Logging file")
    parser.add_argument("--output_file", type=str, default="Nipah_pre_explanations_GPT4A1.csv" , help="Output csv file name")
    parser.add_argument("--log_path", type=str, default='./data_preparation/output/datasets/Q2_wetlab/Nipah/explanation_analysis.log', help="Logging file")

    args = parser.parse_args()
    return args


def evaluate_explanations(data, save_col, RESULT_FOLDER, logger):
    # Log the label distribution
    logger.info("Label distribution")
    logger.info(get_ans_percentage(data, 'Label'))

    # Generate a cross tabulation of the mapped column and 'Review'
    cross_tab = pd.crosstab(data[f"{save_col}_TF"], data["Review"])
    logger.info(f"Cross Table for {save_col}_TF and Review:\n{cross_tab}")

    # Initialize a ModelEvaluator to evaluate the model's performance
    test_evaluator = ModelEvaluator(
        output_path= RESULT_FOLDER,
        title=f"{save_col}_TF",
        calculate_metrics_flag=True,
        plot_confusion_matrix_flag=True,
        save_classification_report_flag=True)

    # Run the model evaluation and get the metrics
    metrics, _ = test_evaluator.run_all(y_true=data['Label'].values, y_pred=data[f"{save_col}_TF"].values,
                                        labels_names=["No", "Yes"])
    logger.info(f"metrics for {save_col}_TF:\n {metrics}")
    return metrics







if __name__ == "__main__":
    args = parse_args()

    # Create the log directory if it doesn't exist and set up the logger
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logger = setup_logger(args.log_path)

    # Parse the JSON formatted file dictionary
    print(args.explanation_dict)
    with open(args.explanation_dict, 'r') as f:
        explanation_dict = json.load(f)
    print(explanation_dict)

    # Set up the result folder
    RESULT_FOLDER = os.path.join(args.input_dir, 'explanations')
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    # Preprocess the combined explanations
    data = pd.read_csv(os.path.join(args.input_dir, args.output_file))
    data = data[data['Review_Paper']=='No'].copy()

    # For each column, evaluate the explanations and save the metrics
    for save_col in explanation_dict.keys():
        # Evaluate explanations for the current column
        metrics = evaluate_explanations(data, save_col, RESULT_FOLDER, logger)
        # Convert the metrics to a DataFrame
        metrics_df = pd.DataFrame([metrics])
        # Save the metrics to a CSV file
        metrics_df.to_csv(os.path.join(RESULT_FOLDER, f'{save_col}_metrics.csv'))

