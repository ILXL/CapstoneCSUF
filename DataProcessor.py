"""
Create CSV files based on criteria
"""


import sys
sys.path.insert(0, './Modules')

import numpy as np
import pandas as pd

from DataHelper import *

STUDENT_DATASET = pd.read_csv("./Datasets/StudentData_121.csv")


PERCENTAGE_COLUMN_NAMES = splitKeywords(
    STUDENT_DATASET, "Percentage")
PERCENTAGE_DATA = STUDENT_DATASET[PERCENTAGE_COLUMN_NAMES]
PERCENTAGE_DATA = cleanDataset(PERCENTAGE_DATA)
PERCENTAGE_DATA = cleanDataName(PERCENTAGE_DATA, readable=False)


###########################
######### Quizzes #########
###########################
save_folder = './Project_Data/QuizMidterms/'

# Clean dataset and name
quiz_column_names = ["Quiz{}".format(x) for x in range(
                    1, 13)] + ['Midtermexamtotal', 'Finalexamtotal']

quiz_data = PERCENTAGE_DATA[quiz_column_names]

showStudentCorrelation(quiz_data, save_path=save_folder + 'CorrelationMatrix.png')
showStudentGradeHeatMap(quiz_data, save_path=save_folder + 'InitialHeatmap.png')
createHistogram(quiz_data, save_path=save_folder+'RealStudentHistogram')

label = np.full_like(len(quiz_data), 1)
quiz_data = quiz_data.assign(real=label)
quiz_data.to_csv("./Processed_Data/QuizMidtermData.csv", index=False)




############################
### Correlation Features ###
############################
save_folder = './Project_Data/CorrelationFeatures/'
high_correlation_features, low_correlation_featres = getHighestCorrFeatures(PERCENTAGE_DATA)

# Load data from percentage dataset with these columns
high_corr_data = PERCENTAGE_DATA[high_correlation_features]
low_corr_data = PERCENTAGE_DATA[low_correlation_featres]
# Create initial graphs
showStudentCorrelation(low_corr_data, save_path=save_folder + 'LowCorrelationMatrix.png')
showStudentCorrelation(high_corr_data, save_path=save_folder + 'HighCorrelationMatrix.png')
showStudentGradeHeatMap(high_corr_data, save_path=save_folder + 'InitialHeatmap.png')
createHistogram(high_corr_data, save_path=save_folder+'RealStudentHistogram')

label = np.full_like(len(high_corr_data), 1)
high_corr_data = high_corr_data.assign(real=label)
high_corr_data.to_csv("./Processed_Data/CleanCorrData.csv", index=False)

low_corr_data = low_corr_data.assign(real=label)
low_corr_data.to_csv("./Processed_Data/CleanLowCorrData.csv", index=False)

############################
### All features ###
############################
save_folder = './Project_Data/AllAssignments/'

# Create initial graphs
showStudentCorrelation(PERCENTAGE_DATA, save_path=save_folder + 'CorrelationMatrix.png')
showStudentGradeHeatMap(PERCENTAGE_DATA, save_path=save_folder + 'InitialHeatmap.png')
createHistogram(PERCENTAGE_DATA, save_path=save_folder+'RealStudentHistogram')

label = np.full_like(len(PERCENTAGE_DATA), 1)
PERCENTAGE_DATA = PERCENTAGE_DATA.assign(real=label)
PERCENTAGE_DATA.to_csv("./Processed_Data/PERCENTAGE_DATA.csv", index=False)







