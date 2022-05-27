# TF_RNN
TF NLP Sample


Hello everyone
This is a little demo on how to train an NLP functioning model.

After cloning and be special aware of the version of the packages needed to install (check requirement file),
here is a list of step on how to use it.

First:
  - We'll create two lists of sentences each in a .txt file. Each sentence must be separated by a carriage return and end in a blank space:
  1. a control list composed by random sentences so we can distinguish between the semantic field and any other sentence. (e.g. control_file_sentences.txt)
  2. a sentence list of the semantic field (say you want to predict whether a sentence is a job offer, or if the sentence denotes stress, or ailments, etc) (e.g. job_offer_sentences.txt)


Second:
  - We'll copy the two text file in our /dictionaries/ folder.
  - With the 0_Dict_to_csv.py we'll clean the files and make a .csv Library
  - we just need to replace the names of the script with the text files we previously created:
  - line 7: dict_name_control = "control_file_sentences.txt"
  - ine 8: dict_name_indicator = "job_offer_sentences.txt"
  - line 9: CSV_FILE_NAME = "Libraries/dic_control-job_offers.csv"
    
Third: 
  - with 1_CSV_to_TF_Training.py we can configure our neural network (Recomend that bidirectional LSTM is an efficent way to harness context out of sentences).
  - We'll also need to do some settings:
  - line 39: modelsaving_path = os.path.join(cwd, "SavedModelMyModel") - write the name of our model: e.g. SavedModelMyModel
  - line 40: dic_csv_path = os.path.join(cwd, "Libraries/dic_control-job_offers.csv") - write the name of the library we created in step two: e.g. Libraries/dic_control-job_offers.csv
  - A new folder we'll be created with the model we'd just train. A plot with the accuracy/loss graphs we'll be displayed and can be saved.
  
Fourth:
  - We can test our model in 2_Restore_Saved_Model.py
  - just neet to replace our model name:
  - line 7: modelsaving_path = os.path.join(cwd, "SavedModelMyModel")
  - and write some sample text to try:
  - line 10 : sample_text = ['I'm going to have a sandwich', 'what's your job experience ?']
  - Control tag will appear for control labels and indicator tag will appear for indicator label:
  - frase:   I'm going to have a sandwich  ---> Resultado: [Control]
  - frase:   what's your job experience ? ---> Resultado: [Indicator]
  
  
  
