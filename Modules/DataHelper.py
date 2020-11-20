# GAN NEURAL NETWORK LIBRARY
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from math import ceil


def truncate(name):
    """Limits the number of characters within a string to 20 characters
    
    Args:
        name 
    
    """
    if len(name) > 20:
        return name[:21]
    return name
    

def getFeatures(columnList, *args):
    """Retrieve column names from a Pandas DataFrame based on a criteria
    
    Example:
        Retrieve all quizzes - getFeatures(DataFrame, 'Quiz')
        Returns all column names within DataFrame that has Quiz in the name
    
    Args:
        columnList (DataFrame): column values
        args (str): Keywords to look for in column values

    Returns:
        Numpy array of column names that match args keywords
    """
    if len(args) == 0:
        return np.array([k for k in dataset.columns.values if k != 'real'])
    else:
        features = set()
        for columnName in columnList.columns.values:
            for keyword in args:
                if keyword.lower() in columnName.lower():
                    features.add(columnName)

        return np.array(list(features))

def showPerformance(dataset: pd.DataFrame, title: str, save_path='./Project_Data/StudentGradeHeatMap.png'):
    """Display line plot of average performance among features within a DataFrame
    
    Note: Limited to 14 features for best visual experience

    Args:
        dataset (DataFrame): Student Grades for assignments
        title (str): Title of plot to be generated
        save_path (str): Save path for plot image

    Returns:
        Plot of average grades for assignments (columns) from a dataset
    """
    
    
    if len(dataset) != 0:
        plt.close()
        size = min(14, len(dataset.columns.values))
        
        class_assignment_names = list(map(truncate, dataset.columns.values[:size]))
        class_assignments = dataset[dataset.columns.values[:size]]
        
        plt.title(title)
        plt.plot(class_assignment_names, class_assignments.mean(axis=0))
        plt.xticks(class_assignment_names, rotation=45, ha="right",
             rotation_mode="anchor")
        plt.yticks(range(0,101,10))
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()

def showPerformanceOverlap(dataset1: pd.DataFrame, dataset2: pd.DataFrame, title: str, save_path='./Project_Data/StudentGradeHeatMap.png'):
    """Display line plots of average performance among features for two DataFrames

    Visualize how two DataFrames are different from each other
    
    Note: Limited to 14 features for best visual experience

    Args:
        dataset1 (DataFrame): Student Grades for assignments
        dataset2 (DataFrame): Student Grades for assignments
        title (str): Title of plot to be generated
        save_path (str): Save path for plot image

    Returns:
        Plot of average grades for assignments (columns) from a dataset
    """
    
    if len(dataset1) != 0:
        plt.close()
        size = min(14, len(dataset1.columns.values))
        
        columns = dataset1.columns.values[:size]
        
        dataset1 = dataset1[columns]
        dataset2 = dataset2[columns]
        
        truncated_names = list(map(truncate, dataset1.columns.values))
        plt.title(title)
        plt.plot(truncated_names, dataset1.mean(axis=0),c='red', label='Real')
        plt.plot(truncated_names, dataset2.mean(axis=0),c='blue', label='Generated', linestyle='dashed')
        plt.legend()
        plt.xticks(truncated_names, rotation=45, ha="right",
             rotation_mode="anchor")
        plt.yticks(range(0,101,10))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def showStudentGradeHeatMap(dataset: pd.DataFrame, save=True, save_path='./Project_Data/StudentGradeHeatMap.png', title="Student Grades Over a Semester"):
    """Display heatmap of random students within a DataFrame

    Note: Limited to 14 features for best visual experience

    Args:
        dataset (DataFrame): Student Grades for assignments
        save (bool): Determine if plot should be saved or just s hown
        save_path (str): Save path for plot image
        title (str): Title of plot to be generated
    
    Returns:
        Heatmap of student grades

    Credit: Matplotlib.org for majority of logic for the heatmap
    """
    plt.close()

    features = dataset.columns.values
    dataset = dataset.to_numpy()

    number_students = 15
    number_assignments = min(14,  len(features))
    
    features = list(map(truncate, features))[:number_assignments]
    plt.close()
    fig, ax = plt.subplots()
    im = ax.imshow(dataset[:number_students, :number_assignments], aspect='auto')

    
    # We want to show all ticks...
    ax.set_xticks(np.arange(number_assignments))
    ax.set_yticks(np.arange(number_students))
    # ... and label them with the respective list entries
    ax.set_xticklabels(features)

    students = ["Student {0}".format(k+1) for k in range(0, number_students)]
    ax.set_yticklabels(students)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # NOTE: Uncomment below to put numbers on individual grid
    # Loop over data dimensions and create text annotations.
    # for i in range(len(students)):
    # 	for j in range(len(features)):
    # 		text = ax.text(j, i, dataset[i, j],
    # 					ha="center", va="center", color="w")

    ax.figure.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.close()


def showStudentCorrelation(dataset: pd.DataFrame, save=True, save_path='./Project_Data/CorrelationMatrix.png'):
    """
    Display correlation values of features within a dataframe

    Note: Limited to 15 features for best visual experience

    Args:
        dataset (DataFrame): Student grades and assignments
        save (bool): Determine if plot should be saved or just s hown
        save_path (str): Save path for plot image
        title (str): Title of plot to be generated
    
    Returns:
        Correlation matrix of features within a dataframe

    Credit: Matplotlib.org for majority of logic for the heatmap
    """

    plt.close()
    number_students = min(15, len(dataset.columns.values))
    dataset = dataset[dataset.columns.values[:number_students]].corr(method="pearson")
    column_vals = list(map(truncate, dataset.columns.values))

    fig, ax = plt.subplots()
    im = ax.imshow(dataset, aspect='auto', cmap='YlGn')

    # We want to show all ticks...
    ax.set_xticks(np.arange(number_students))
    ax.set_yticks(np.arange(number_students))
    # ... and label them with the respective list entries
    ax.set_xticklabels(column_vals)
    ax.set_yticklabels(column_vals)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # NOTE: Uncomment below to put numbers on individual grid
    # Loop over data dimensions and create text annotations.
    grades = dataset.to_numpy()
    for i in range(number_students):
        for j in range(number_students):
            text = ax.text(j, i, round(grades[i, j], 1),
                           ha="center", va="center", color="black")

    ax.figure.colorbar(im, ax=ax)

    ax.set_title("Correlation Matrix")
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.close()


def splitKeywords(dataframe, *args):
    """Gathers all columns that match an args value
    
    Example:
        splitKeywords(Data, 'Real', 'Percentage')
        Returns: [['Real1', 'Real2'..], ['Percentage1', 'Percentage2'..]]

    Args:
        dataset (DataFrame): Student grades and assignments
        args (str): Keywords to look for in column to extract
    
    Returns:
        Returns groups of columns with arg value keywords in it

    """
    dataset_splits = {}
    for kw in args:
        if dataset_splits.get(kw) == None:
            dataset_splits[kw] = []
        try:
            dataset_splits[kw] = [col for col in dataframe if (
                kw in col and len(re.findall(r"\)\.1", col)) == 0)]
            if len(dataset_splits[kw]) == 0:
                print("No column named", kw)
                del dataset_splits[kw]
        except:
            pass

    if len(dataset_splits) == 1:
        return dataset_splits[kw]
    return dataset_splits.values()


def cleanDataName(dataset: pd.DataFrame, readable=True):

    def cleanNames(column_name):
        if readable:
            column_name = re.sub(
                r"['',':','(',')']|Real|(Percentage)|Quiz:|Assignment:", "", column_name)
        else:
            column_name = re.sub(
                r"[' ',':','(',')']|Real|(Percentage)|Quiz:|Assignment:", "", column_name)
        return column_name

    dataset = dataset.rename(cleanNames, axis='columns')

    return dataset


def cleanDataset(dataset: pd.DataFrame):
    """Cleans the columns
    
    Removes empty cells and replaces it with a 0 or null keyword
    Columns that contain 25% missing data are automatically dropped


    Args:
        dataset (DataFrame): Student grades and assignments
    
    Returns:
        Returns Dataframe with numerical values
    """

    # Remove a column if its column contains more than 25% empty values
    dataset = dataset.copy()

    for col in dataset:
        value_freq = dataset[col].value_counts().to_dict()
        num_blank = value_freq.get("-")
        num_zero = value_freq.get(0)
        if num_blank != None:
            if num_blank > len(dataset)*.25:
                dataset = dataset.drop(col, axis=1)
        if num_zero != None:
            if num_zero > len(dataset)*.25:
                dataset = dataset.drop(col, axis=1, inplace=True)
    dataset.replace(" %", "", regex=True, inplace=True)

    # Replace percent values into real numbers i.e. 25% ==> 25.0
    for col in dataset.columns.values:
        dataset[col] = dataset[col].apply(pd.to_numeric, errors='coerce')
    
    dataset = dataset.replace(to_replace="-", value=0.0).astype("float64")
    dataset = dataset.fillna(0.0).clip(0,100)

    return dataset


def getHighestCorrFeatures(dataset: pd.DataFrame):

    def create_corrMatrix(dataframe):
        # Create the correlation matrix and strip where all values NAN
        assert(len(dataframe) != None)
        matrix = dataframe.corr(method="pearson")
        np.fill_diagonal(matrix.values, np.nan)
        for x in matrix:
            value_freq = matrix[x].value_counts().to_dict()
            # Drop values if it is empty
            if len(value_freq) == 0:
                matrix.drop(x, axis=1, inplace=True)
                matrix.drop(x, axis=0, inplace=True)

        return matrix

    def find_highest_corr(data):
        highest_corr_labels = []
        for x in data:
            # Get columns with highest correlation values
            large = data[x].sort_values(key=abs, ascending=False)[:3]
            for d in large.iteritems():
                keys = [x, d[0]]
                keys.sort()
                if (keys, d[1]) not in highest_corr_labels:
                    highest_corr_labels.append((keys, d[1]))

        return highest_corr_labels

    def find_lowest_corr(data):
        highest_corr_labels = []
        for x in data:
            # Get columns with highest correlation values
            large = data[x].sort_values(ascending=True)[:3]
            for d in large.iteritems():
                keys = [x, d[0]]
                keys.sort()
                if (keys, d[1]) not in highest_corr_labels:
                    highest_corr_labels.append((keys, d[1]))

        return highest_corr_labels
    
    def max_corr(val):
        return val[1]

    corrMatrix = create_corrMatrix(dataset)
    # Get the labels that have high correlations with other values
    highest_corr_labels = find_highest_corr(corrMatrix)
    highest_corr_labels.sort(key=max_corr, reverse=True)
    highest_corr_labels = highest_corr_labels[:ceil(
        len(highest_corr_labels)*.25)]

    lowest_corr_labels = find_lowest_corr(corrMatrix)
    lowest_corr_labels.sort(key=max_corr, reverse=True)
    lowest_corr_labels = lowest_corr_labels[:ceil(
        len(lowest_corr_labels)*.25)]

    relevant_labels = set()
    for keys in highest_corr_labels:
        for k in keys[0]:
            if abs(keys[1]) > 0.9:
                relevant_labels.add(k)

    neg_relevant_labels = set()
    for keys in lowest_corr_labels:
        for k in keys[0]:
            if keys[1] < 0:
                neg_relevant_labels.add(k)

    return [label for label in dataset.columns.values if label in relevant_labels],  [label for label in dataset.columns.values if label in neg_relevant_labels]

def createHistogram(dataset: pd.DataFrame, save=True, save_path='./Project_Data/RealStudentHistogram.png', title='Histogram of Real Student Grades'):
    tensor = dataset.to_numpy().flatten()
    max_value = 100
    min_value = 0

    plt.clf() 
    plt.title(title)
    plt.xlabel('Grades (%)')
    plt.ylabel('Frequency Density')
    plt.ylim(0,.1)
    plt.hist(tensor, bins=10, histtype='stepfilled', range=(min_value,max_value),color='blue',density=True)
    plt.savefig(save_path)
