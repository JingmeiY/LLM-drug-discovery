#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix,  classification_report)
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from funs import get_output_file,setup_logger




def get_ans_percentage(df, ans_col):
    count = df[ans_col].value_counts()
    percentage = df[ans_col].value_counts(normalize=True) * 100
    result = pd.concat([count, percentage], axis=1)
    result.columns = ['Count', 'Percentage']
    return result






def save_mean_std_metrics(cv_metrics, saved_path, saved_csv):
    # Calculate mean and standard deviation of metrics across all folds for the testing set
    cv_metrics_df = pd.DataFrame(cv_metrics)
    cv_mean_metrics = cv_metrics_df.mean(axis=0).round(2)
    cv_std_metrics = cv_metrics_df.std(axis=0).round(2)
    # Concatenate the DataFrames
    metrics_summary = pd.concat([cv_metrics_df, cv_mean_metrics.to_frame().T, cv_std_metrics.to_frame().T], ignore_index=True)
    # Set the index names
    index_names = list(range(cv_metrics_df.shape[0])) + ['mean', 'std']
    metrics_summary.index = index_names
    # Save the summary of metrics for each label combination
    metrics_summary.to_csv(os.path.join(saved_path, saved_csv), index=True)


class ModelEvaluator:
    def __init__(self, output_path, title,
                 # average='weighted',
                 calculate_metrics_flag=True,
                 plot_confusion_matrix_flag=True,
                 save_classification_report_flag=True):
        # self.average = average
        self.output_path = output_path
        self.title = title
        self.calculate_metrics_flag = calculate_metrics_flag
        self.plot_confusion_matrix_flag = plot_confusion_matrix_flag
        self.save_classification_report_flag = save_classification_report_flag


    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0) * 100
        f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0) * 100
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        cm = confusion_matrix(y_true, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100

        return {
            'Accuracy': round(accuracy, 2),
            'Sensitivity': round(recall, 2),
            'F1_binary': round(f1_binary, 2),
            'Precision': round(precision, 2),
            'Specificity': round(specificity, 2),
            'F1_weighted': round(f1_weighted, 2),
            'F1_macro': round(f1_macro, 2) }

    def save_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, zero_division=0)
        with open(os.path.join(self.output_path, f"{self.title}_classification_report.txt"), "w") as f:
            f.write(f"{self.title}\n\n{report}")
        print(report)

    def plot_confusion_matrix(self, y_true, y_pred, labels_names):
        """
        Plot the confusion matrix with True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
        Args:
            y_true (list or numpy array): True labels.
            y_pred (list or numpy array): Predicted labels.
            labels_names (list): List of class names for the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()

        # Calculate normalized values
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create a new array for annotating the heatmap with TP, TN, FP, and FN labels along with raw counts and normalized values
        cm_labels = np.empty_like(cm, dtype='<U20')
        cm_labels[0, 0] = f'TN: {cm[0, 0]} ({cm_normalized[0, 0]:.2%})'
        cm_labels[0, 1] = f'FP: {cm[0, 1]} ({cm_normalized[0, 1]:.2%})'
        cm_labels[1, 0] = f'FN: {cm[1, 0]} ({cm_normalized[1, 0]:.2%})'
        cm_labels[1, 1] = f'TP: {cm[1, 1]} ({cm_normalized[1, 1]:.2%})'

        sns.heatmap(cm, annot=cm_labels, fmt='', xticklabels=labels_names, yticklabels=labels_names, cmap="Blues")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(f"Confusion Matrix - {self.title}")
        plt.savefig(os.path.join(self.output_path, f"{self.title}_confusion_matrix.jpg"))
        plt.close()
        return cm

    def run_all(self, y_true, y_pred, labels_names):
        if self.calculate_metrics_flag:
            metrics = self.calculate_metrics(y_true, y_pred)
            print("Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

        if self.save_classification_report_flag:
            self.save_classification_report(y_true, y_pred)

        if self.plot_confusion_matrix_flag:
            cm = self.plot_confusion_matrix(y_true, y_pred, labels_names)
        return metrics, cm

def plot_mean_confusion_matrix(cm, title, labels_names, saved_path, saved_figure):

    plt.figure()
    # Calculate normalized values
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a new array for annotating the heatmap with TP, TN, FP, and FN labels along with raw counts and normalized values
    cm_labels = np.empty_like(cm, dtype='<U20')
    cm_labels[0, 0] = f'TN: {cm[0, 0]} ({cm_normalized[0, 0]:.2%})'
    cm_labels[0, 1] = f'FP: {cm[0, 1]} ({cm_normalized[0, 1]:.2%})'
    cm_labels[1, 0] = f'FN: {cm[1, 0]} ({cm_normalized[1, 0]:.2%})'
    cm_labels[1, 1] = f'TP: {cm[1, 1]} ({cm_normalized[1, 1]:.2%})'

    sns.heatmap(cm, annot=cm_labels, fmt='', xticklabels=labels_names, yticklabels=labels_names, cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"Mean Confusion Matrix - {title}")
    plt.savefig(os.path.join(saved_path, saved_figure))
    plt.close()


def parse_args():

    parser=argparse.ArgumentParser(description="Model evaluation of GPT-3",
                                   prog="GPT3",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds used in cross validation")
    parser.add_argument("--answer_path",type=str, default="./models/output/Q2_wetlab/Nipah/cross_validation_output/answer_columns.txt", help = "Path to the answer  names")
    parser.add_argument("--input_path",type=str, default='./models/output/Nipah/cross_validation_output', help = "Path to the testing set")
    parser.add_argument("--output_folder", type=str, default='./models/results/Nipah/cross_validation_output', help="directory to save the results")
    parser.add_argument("--log_path", type=str, default='./models/results/Nipah/cross_validation_output/evaluation.log', help="file name for logging")
    parser.add_argument("--review_paper", type=int, default=1, help="whether to filter review_paper")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = setup_logger(args.log_path)
    NFOLD = args.num_fold

    with open(args.answer_path,'r') as f:
        cols = [line.strip() for line in f.readlines()]
    for answer_col in cols:
        RESULT_FOLDER = os.path.join(args.output_folder, answer_col)
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        SUB_FOLDER = os.path.join(args.input_path, answer_col)

        cv_metrics, cv_confusion_matrices = [], []
        for fold in range(NFOLD):

            ANS_COL = f"{answer_col}_{fold}"
            ANS_TF = f"{ANS_COL}_TF"

            logger.info(f"Evaluating {ANS_COL}!!!!!!!!!!!!!!!!!!")

            df = pd.read_csv(get_output_file(SUB_FOLDER,  f"{ANS_COL}_ans.csv"), encoding='latin-1')
            if not bool(args.review_paper):
                df = df[df['Review_Paper']=='No'].copy()

            logger.info("Label distribution")
            logger.info(get_ans_percentage(df, 'Label'))
            # print(f"The percentage of positive cases in the dataset: {df['Label'].sum() *100/df['Label'].count():.2f}")

            # Evaluate the model on the test set
            test_evaluator = ModelEvaluator(
                output_path=RESULT_FOLDER,
                title= ANS_COL,
                # average= 'binary',
                calculate_metrics_flag=True,
                plot_confusion_matrix_flag=True,
                save_classification_report_flag=True)

            fold_metrics, fold_cm = test_evaluator.run_all(y_true= df['Label'].values, y_pred=df[ANS_TF].values,labels_names=["No", "Yes"])
            cv_confusion_matrices.append(fold_cm)
            cv_metrics.append(fold_metrics)

        save_mean_std_metrics(cv_metrics, saved_path= RESULT_FOLDER, saved_csv=f"{answer_col}_{NFOLD}folds.csv")
        # Calculate and plot the mean confusion matrix for each feature set across 5 folds
        cv_mean_cm = np.mean(cv_confusion_matrices, axis=0)
        plot_mean_confusion_matrix(cm=cv_mean_cm, title = answer_col, labels_names=["No", "Yes"],
                                   saved_path=RESULT_FOLDER, saved_figure=f"{answer_col}_mean_cf.JPG")

if __name__ == '__main__':
    main()


