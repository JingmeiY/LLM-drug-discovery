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
from funs import setup_logger
from parse_subquestion_json import JsonProcessor
from mapping_response import map_text_to_label_cot_string

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



def preprocess_explanations(data, columns):
    # Transform 'Review' to a binary 'Label' based on condition
    data['Label'] = data['Review'].apply(lambda x: 1 if 'YES' in x.upper() else 0)

    for column in columns:
        if 'sub' in column:
            jp = JsonProcessor()
            data[f"{column}_TF"] = data[column].apply(
                lambda x: jp.map_text_to_label_sub_json(x, process_subquestions=False, sub_neg_q=False, sub_pos_q=False, sub_pos_last=False))
        else:
            data[f"{column}_TF"] = data[column].apply(map_text_to_label_cot_string)
    return data



def combine_explanations(file_dict, directory):
    """
    Function to combine multiple explanation files into one DataFrame
    """
    dfs = []
    for col, file in file_dict.items():
        # df = pd.read_csv(os.path.join(directory, file))
        df = pd.read_csv(os.path.join(directory, file), encoding='latin-1')
        dfs.append(df)
    # Concatenate dataframes and remove duplicate columns
    data = pd.concat(dfs, axis=1).loc[:, ~pd.concat(dfs, axis=1).columns.duplicated()]
    return data

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


    # Combine explanations from different files
    combined_data = combine_explanations(explanation_dict, args.input_dir)

    # Preprocess the combined explanations
    data = preprocess_explanations(combined_data, list(explanation_dict.keys()))
    data.to_csv(os.path.join(args.input_dir, args.output_file), index=False)



