import os
import pandas as pd

stimulus = 'S1obj'
# stimulus = 'S2match'
# stimulus = 'S2nomatch'

subjects = []
for file in os.listdir(os.fsencode('Datasets/eeg_full_correlations/')):
    filename = os.fsdecode(file)
    subject = filename[:4]
    if subject not in subjects: subjects.append(subject)
    
node_labels = ['FPZ', 'FP1', 'FP2',
               'AFZ', 'AF1', 'AF2', 'AF7', 'AF8',
               'FZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
               'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FT7', 'FT8',
               'CZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
               'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'TP7', 'TP8',
               'PZ','P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
               'POZ', 'PO1', 'PO2', 'PO7', 'PO8',
               'O1', 'O2', 'OZ']


all_trials = {subject:[] for subject in subjects}
for file in os.listdir(os.fsencode('Datasets/eeg_full_correlations/')):
    filename = os.fsdecode(file)
    subject = filename[:4]
    if filename[4:4+len(stimulus)]==stimulus: 
        df = pd.read_csv(f'Datasets/eeg_full_correlations/{filename}', delimiter=' ', usecols=[1, 2, 3])
        S = np.nan_to_num(nx.to_numpy_array(nx.from_pandas_edgelist(df, source='sensor1', target='sensor2', edge_attr='weight'), nodelist=node_labels), nan=0.0)
        if np.linalg.norm(S) > 0.: all_trials[subject].append(S)
dataset_61 = np.array([np.abs(np.mean(np.array(all_trials[subject]), axis=0)) for subject in subjects])
group_labels = np.array([subject[0] for subject in subjects])

with open(f'Datasets/eeg_{stimulus}-dataset.npy', 'wb') as file: np.save(file, dataset_61)
with open(f'Datasets/eeg_{stimulus}-group_labels.npy', 'wb') as file: np.save(file, group_labels)
