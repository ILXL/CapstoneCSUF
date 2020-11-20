import sys
sys.path.insert(0, './Modules')

from GAN import GAN
from NeuralNetworks import *
from DataHelper import *
import scipy.io as scp
import numpy as np
import pandas as pd
import json

with open('./DataInformation.json') as f:
    config = json.load(f)

currentData = config['Quizzes']


hyperparameters = currentData['Hyperparameters']
dataFile = currentData['DataPath']
folder = currentData['SaveFolderName']


# Load dataset and set up features
education_data = pd.read_csv(
    dataFile, index_col=False)

GAN_NN = GAN(filepath=dataFile)

# Initialize models for the GAN
D_Network = RNNDiscriminator(education_data)
G_Network = generatorModelModified(education_data)
GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)

# Generate grades and graphs from untrained model
untrained_generation = GAN_NN.generateFakeData(size=1)
print("Initial generation\n", untrained_generation)
createHistogram(untrained_generation, save_path=folder + 'InitialGeneratedStudentHistogram.png', title='Histogram of Initial Generated Student Grades')
showStudentGradeHeatMap(education_data, save=True,
                        save_path=folder + 'GeneratedHeatmap.png',  
                        title="Generated Student Grades Over a Semester")
showPerformance(education_data, 'Real Student Performance', save_path=folder + 'RealStudentPerformance.png')



# Train network
print("Training Network...")
test = GAN_NN.train_network(epochs=hyperparameters['Epochs'], 
                            batch_size=hyperparameters['Batch Size'], 
                            history_steps=hyperparameters['Checkpoint Frequency'],
                            checkpoint_path=currentData['CheckpointPath'])
print("Finished Training, creating histogram")


# Create data from trained models
while True:
    try:
        trained_generation = GAN_NN.generateFakeData(size=1)
        GAN_NN.saveLossHistory(folder + 'LossHistory')
        
        # Create generated student class and save
        print("Final generation\n", trained_generation)
        generated_class = GAN_NN.generateFakeData(size=len(education_data))
        generated_class.to_csv(folder + 'GeneratedData.csv')

        # Create graphs and performance of trained network
        showStudentGradeHeatMap(generated_class, save=True,
                                save_path=folder + 'GeneratedHeatmap.png',  
                                title="Generated Student Grades Over a Semester")
        createHistogram(generated_class, save_path=folder + 'GeneratedStudentHistogram.png', title='Histogram of Generated Student Grades')
        showPerformance(generated_class, 'Generated Student Performance', save_path=folder + 'GeneratedStudentPerformance.png')
        showPerformanceOverlap(education_data, generated_class, 'Class Average Performance',  save_path=folder + 'ClassPerformance.png')
        break
    except Exception as e:
        print(e)
        print('Make sure files are closed')
        s = input('Press Enter to Continue')




