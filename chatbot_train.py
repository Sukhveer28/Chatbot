from chatterbot import ChatBot
from chatterbot.trainers import ListTrainers
import os

def setup():
    chatbot = ChatBot('Bot')

    trainers = ListTrainer

    training_data_dir = '/Users/priyanshityagi/Documents/chatbot'
    target_file = 'traininng_data.csv'
    if target_file in os.listdir(training_data_dir):
        file_path == os.path.join(training_data_dir, target_file)
        with open(file_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()
        trainer.train(lines)

        print(f"Training completed with data from {target_file}")
    else:
        print(f"File '{target_file}' not found in the directory.")


