from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import sys
import os
from src.logger import logging 
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("models", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df, test_df):
        try:
            logging.info("Loading the data for the training and testing.")
            train_data = train_df # Reading Training Data
            test_data = test_df # Reading Testing Data
            x_train, y_train, x_test, y_test = train_data.iloc[:, 0], train_data.iloc[:, 1], test_data.iloc[:, 0], test_data.iloc[:, 1] # Split labels and features
            
            x_train = x_train.astype('str')
            x_test = x_test.astype('str')
            voc_size = 10000 # Size of the vocabulary

            # Tokenization and padding
            tokenizer = Tokenizer(num_words=voc_size)
            tokenizer.fit_on_texts(x_train)
            sequences_train = tokenizer.texts_to_sequences(x_train)
            sequences_test = tokenizer.texts_to_sequences(x_test)

            max_length = max(len(seq) for seq in sequences_train) # Maximum length for padding

            embedded_docs_train = pad_sequences(sequences_train, maxlen=max_length) # Padding training data
            embedded_docs_test = pad_sequences(sequences_test, maxlen=max_length) # Padding testing data
            
            # Print shapes for debugging
            print("Shape of embedded_docs_train:", embedded_docs_train.shape)
            print("Shape of embedded_docs_test:", embedded_docs_test.shape)

            # One-hot encoding the labels for multi-class classification
            y_train = to_categorical(y_train, num_classes=3)
            y_test = to_categorical(y_test, num_classes=3)
            
            # Print shapes of labels for debugging
            print("Shape of y_train:", y_train.shape)
            print("Shape of y_test:", y_test.shape)
            
            ## Defining the Tensorflow model for sentiment analysis
            model = Sequential()
            model.add(Embedding(input_dim=voc_size, output_dim=128))
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(64)))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='softmax'))  # Use softmax for multi-class classification
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logging.info("Model has been created.")
            logging.info("Model is in training phase.")
            model.fit(embedded_docs_train, y_train, epochs=10, batch_size=32, validation_split=0.3) ## Training the model and splits for validation during training.
            preds = model.predict(embedded_docs_test) ## Predictions
            preds = np.argmax(preds, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            logging.info("Model has been saved.")
            os.makedirs(os.path.dirname(self.model_trainer_config.train_model_file_path), exist_ok=True)
            model.save(self.model_trainer_config.train_model_file_path)
            return accuracy_score(y_test_labels, preds)
        
        except Exception as e:
            raise CustomException(e, sys)
