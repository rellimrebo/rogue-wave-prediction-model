import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import keras

# Helper functions
def normalize_segment(segment):
    '''
    Normalizes truncated segment of data with respect to the maximum absolute value
    Returns the same array if the max is 0
    
    Inputs:
    segment: list containing floats
    
    Outputs:
    segment: normalized list containing floats
    '''
    max_val = max(abs(x) for x in segment)

    if max_val != 0:
        return [x / max_val for x in segment]
    return segment

def normalize_segment_custom(segment, start, end):
    '''
    Normalizes a segment of data within specified start and end points with respect to the maximum value
    Returns the same array if the max is 0
    
    Inputs:
    segment: list containing floats
    start: integer, start index of the segment
    end: integer, end index of the segment
    
    Outputs:
    segment: normalized list containing floats
    '''
    segment = segment[start:end]  # Slice the segment based on start and end points
    max_val = max(abs(x) for x in segment)

    if max_val != 0:
        return [x / max_val for x in segment]
    return segment


# load data
rogue_directory = 'data/rogue'
rogue_data_df = pd.DataFrame()
for filename in os.listdir(rogue_directory):
    if filename.startswith('rogue_wave_data_station') and filename.endswith('.parquet'):
        file_path = os.path.join(rogue_directory, filename)
        df = pd.read_parquet(file_path)
        if 'Segment' in df.columns:
            df['Segment'] = df['Segment'].apply(lambda x: x[:int(28 * 60 * 1.28)])
            rogue_data_df = rogue_data_df.append(df, ignore_index=True)

# # Include data from data/Rogue Wave Data
# additional_rogue_directory = 'data/Rogue Wave Data'
# for filename in os.listdir(additional_rogue_directory):
#     if filename.startswith('rogue_wave_data_station') and filename.endswith('.parquet'):
#         file_path = os.path.join(additional_rogue_directory, filename)
#         df = pd.read_parquet(file_path)
#         if 'Segment' in df.columns:
#             df['Segment'] = df['Segment'].apply(lambda x: x[:int(28 * 60 * 1.28)])
#             rogue_data_df = rogue_data_df.append(df, ignore_index=True)

# Finalize the rogue data DataFrame
rogue_data_df = rogue_data_df[['Segment']]

non_directory = 'data/non rogue'
non_data_df = pd.DataFrame()
for filename in os.listdir(non_directory):
    if filename.startswith('non_rogue_wave_data_station') and filename.endswith('.parquet'):
        file_path = os.path.join(non_directory, filename)
        df = pd.read_parquet(file_path)
        if 'Segment' in df.columns:
            df['Segment'] = df['Segment'].apply(lambda x: x[:int(28 * 60 * 1.28)])
            non_data_df = non_data_df.append(df, ignore_index=True)

# Include data from data/Non Rogue Wave Data
# additional_non_directory = 'data/Non Rogue Wave Data'
# for filename in os.listdir(additional_non_directory):
#     if filename.startswith('non_rogue_wave_data_station') and filename.endswith('.parquet'):
#         file_path = os.path.join(additional_non_directory, filename)
#         df = pd.read_parquet(file_path)
#         if 'Segment' in df.columns:
#             df['Segment'] = df['Segment'].apply(lambda x: x[:int(28 * 60 * 1.28)])
#             non_data_df = non_data_df.append(df, ignore_index=True)

# Finalize the non-rogue data DataFrame
non_data_df = non_data_df[['Segment']]

# Normalize data
# rogue_data_df['Segment'] = rogue_data_df['Segment'].apply(normalize_segment)
# non_data_df['Segment'] = non_data_df['Segment'].apply(normalize_segment)

# Update the normalization process for rogue and non-rogue data
rogue_data_df['Segment'] = rogue_data_df['Segment'].apply(lambda x: normalize_segment_custom(x, int(0*1.28*60), int(24*1.28*60)))
non_data_df['Segment'] = non_data_df['Segment'].apply(lambda x: normalize_segment_custom(x, int(0*1.28*60), int(24*1.28*60)))

# Prepare for LSTM input
X_rogue = np.array(rogue_data_df['Segment'].tolist()).reshape(-1, int(24*1.28*60) - int(0*1.28*60), 1)
X_non = np.array(non_data_df['Segment'].tolist()).reshape(-1, int(24*1.28*60) - int(0*1.28*60), 1)

# Assuming y_rogue and y_non are initially binary labels (0s and 1s)
y_rogue = np.ones(len(X_rogue))
y_non = np.zeros(len(X_non))

# One-hot encode the labels
y_rogue_one_hot = keras.utils.to_categorical(y_rogue, num_classes=2)
y_non_one_hot = keras.utils.to_categorical(y_non, num_classes=2)

# Combine
X = np.concatenate((X_rogue, X_non), axis = 0)
y = np.concatenate((y_rogue_one_hot, y_non_one_hot), axis=0)

# Split into training/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42) # change state here

# Hyperparameters
N_LSTM = 50 # original: 10
p_D = 0.2 # original: 0.05
N_L = 5 # original 4

model = keras.models.Sequential()

# Adding N_L LSTM blocks
for i in range(N_L):
    # LSTM layer
    if i == 0:
        # First layer requires input shape
        model.add(keras.layers.LSTM(N_LSTM, return_sequences=True if i < N_L - 1 else False, input_shape=(int(24*1.28*60) - int(0*1.28*60), 1)))
    else:
        model.add(keras.layers.LSTM(N_LSTM, return_sequences=True if i < N_L - 1 else False))

    # Batch normalization layer
    model.add(keras.layers.BatchNormalization())

    # Dropout layer
    model.add(keras.layers.Dropout(p_D))

    # Fully connected layer
    N_f = 50 if i == 0 else 2  # N_f is 50 for the first stack, 2 for the others
    model.add(keras.layers.Dense(N_f, activation='relu'))

# Output layer for binary classification (rogue wave or not)
model.add(keras.layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary to check if everything is set up correctly
model.summary()

# Now, train the model
history = model.fit(X_train, y_train, 
                    epochs=100,  # Adjust the number of epochs as needed
                    batch_size=128,  # Adjust the batch size as needed
                    validation_split=0.2)  # Using part of the training data for validation

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Plot accuracy vs epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Save the model
model.save('rogue_wave_prediction_model.h5')

# Predict the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()