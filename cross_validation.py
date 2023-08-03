#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

class CrossValidation:
    """
    Class for performing n-fold stratified cross-validation on a pandas dataframe.

    Args:
        n_splits (int, optional): Number of folds to split the data into. Defaults to 5.
        save_train_test_split (bool, optional): Whether to save each train-test split to a file. Defaults to True.
        save_path (str, optional): Directory to save train-test splits in. Defaults to 'cross_validation_datasets'.
        random_state (int, optional): Seed for the random number generator. Defaults to RANDOM_STATE (a constant).


    Methods:
        save_splits(splits, save_split_file): Saves the splits to a pickle file.
        load_splits(save_split_file): Loads the splits from a pickle file.
        get_label_percentage(df, label_col='Review'): Returns the percentage of positive and negative labels in a dataframe.
        split(df, label_col='Review', save_split_file='cross_validation_splits.pkl'): Performs n-fold cross-validation on a dataframe.
    """
    def __init__(self, n_splits=5,
                 save_train_test_split = True, random_state= RANDOM_STATE,
                 save_path='cross_validation_datasets'):
        self.n_splits = n_splits
        self.save_path = save_path
        self.random_state= random_state
        self.save_train_test_split = save_train_test_split
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_splits(self, splits, save_split_file):
        """
            Saves the splits to a pickle file.

            Args:
                splits (list): List of tuples containing train and test indices, and label percentages for each split.
                save_split_file (str): File name to save the pickle file as.
        """

        with open(os.path.join(self.save_path, save_split_file), 'wb') as f:
            pickle.dump(splits, f)

    def load_splits(self, save_split_file):
        """
            Loads the splits from a pickle file.

            Args:
                save_split_file (str): File name of the pickle file to load.

            Returns:
                list: List of tuples containing train and test indices, and label percentages for each split.
        """
        with open(os.path.join(self.save_path, save_split_file), 'rb') as f:
            splits = pickle.load(f)
        return splits

    def get_label_percentage(self, df, label_col='Review'):
        """
        Returns the percentage of positive and negative labels in a dataframe.

        Args:
            df (pandas dataframe): Dataframe to calculate label percentages for.
            label_col (str, optional): Name of the column containing labels. Defaults to 'Review'.

        Returns:
            pandas dataframe: Dataframe containing the count and percentage of positive and negative labels.
        """
        count = df[label_col].value_counts()
        percentage = df[label_col].value_counts(normalize=True) * 100
        result = pd.concat([count, percentage], axis=1).round(2)
        result.columns = ['Count', 'Percentage']
        return result


    def split(self, df,
              label_col='Review',
              save_split_file='cross_validation_splits.pkl'):
        """
        This function split performs the stratified k-fold cross-validation on a given dataframe and saves the train-test splits into CSV files, if save_train_test_split argument is set to True.
        Parameters:
        df : pandas DataFrame
        Input dataframe for cross-validation.

        label_col : str, optional (default='Review')
        The column name of target variable.

        save_split_file : str, optional (default='cross_validation_splits.pkl')
        The filename for pickle file to save train-test splits.

        Returns:
        splits : list
        A list of tuples. Each tuple contains:
        train index
        test index
        """
        splits = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        label_distributions = {}  # To keep track of label distributions for all folds

        for i, (train_index, test_index) in enumerate(skf.split(df, df[label_col])):
            train_df = df.loc[train_index].copy()
            test_df = df.loc[test_index].copy()
            train_label_percentage = self.get_label_percentage(train_df)
            label_distributions[f'train_{i}'] = train_label_percentage
            test_label_percentage = self.get_label_percentage(test_df)
            label_distributions[f'test_{i}'] = test_label_percentage
            if self.save_train_test_split:
                train_filename = f'{self.save_path}/train_{i}.csv'
                test_filename = f'{self.save_path}/test_{i}.csv'
                train_df.to_csv(train_filename, index = False)
                test_df.to_csv(test_filename,  index = False)
            splits.append((train_index, test_index))

        # Save the train and test indices for each fold
        self.save_splits(splits, save_split_file)

        # Concatenate all the label distribution dataframes and save to a csv file
        label_dist_df = pd.concat(label_distributions, names=['Dataset', 'Label'])
        label_dist_df.to_csv(os.path.join(self.save_path, 'label_distribution.csv'))

        return splits

