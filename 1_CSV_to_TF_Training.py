import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def variableTypeCheck(pd_dictionary_train_fn):
    for index, row in pd_dictionary_train_fn.iterrows():
        if isinstance(row[1], str) == True:
            pass
        else:
            print("string: not ok", index, row[1])
            pd_dictionary_train_fn = pd_dictionary_train_fn.drop(index)

        if isinstance(row[0], int) == True:
            pass
        else:
            print("int: not ok", row[0], row[1])

    return pd_dictionary_train_fn


################# INICIO

# "SavedModelMED" "Libraries/dic_control-medicamentos.csv"
# "SavedModelEstres" "Libraries/dic_control-estres.csv"
# SavedModelQuejaSomatica Libraries/dic_control-Quejas_somaticas.csv
# Libraries/dic_control-somatic_meta.csv SavedModelSomaticMeta
cwd = os.getcwd()
# modelsaving_path = os.path.join(cwd, "SavedModelSomaticMeta")
# dic_csv_path = os.path.join(cwd, "Libraries/dic_control-somatic_meta.csv")
modelsaving_path = os.path.join(cwd, "SavedModelIngredientes")
dic_csv_path = os.path.join(cwd, "Libraries/dic_control-ingredientes.csv")

dictionary_train = pd.read_csv(dic_csv_path,
    names=["label", "text"], header=None , encoding = 'utf-8') # encoding = 'windows-1252')
#  , encoding='utf-8-sig'

pd_dictionary_train = pd.DataFrame(dictionary_train)

pd_dictionary_train_clean = variableTypeCheck(pd_dictionary_train)

for row in pd_dictionary_train_clean:
    print(row)

training_df: pd.DataFrame = pd_dictionary_train_clean

features = [ 'text']

print("-------------------",len(training_df),training_df['text'].describe())

dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(training_df['text'].values, tf.string),
            tf.cast(training_df['label'].values, tf.int32)
        )
    )
)
train_size = int(round(len(training_df)*0.8))
full_dataset = dataset.shuffle(buffer_size=10000)
training_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
encoder.adapt(test_dataset.map(lambda text, label: text))
# print(encoder.get_vocabulary())
vocab = np.array(encoder.get_vocabulary())

from tensorflow.python import keras
model = keras.Sequential([
    encoder,
    keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    keras.layers.Bidirectional(keras.layers.LSTM(256)), #64
    keras.layers.Dense(128, activation='relu'), # 64
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

# model.fit(train_dataset, epochs=90, # 50 # also 25 med : 100
#                     validation_data=test_dataset,
#                     validation_steps= 10) # 15 # also 30 med : 250
history = model.fit(train_dataset, epochs=18, # 50 # also 25 med : 100
                    validation_data=test_dataset,
                    validation_steps= 64) # 15 # also 30 med : 250


model.save(modelsaving_path)

# tester =testingNet(encoder, train_dataset, test_dataset) # OJO aquí el testing
# predict_sample(model) # OJO aquí predicciones
sample_text = "hola que tal"
# predict(model, sample_text)



plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()




# 2. frase:  no estoy cansado  ---> Resultado: [Control]
# 2. frase:  me siento bien  ---> Resultado: [Control]
# 1. frase:  para nada estoy bien  ---> Resultado: [Control]
# 2. frase:  es que no doy mas de si, al limite de mis fuerzas  ---> Resultado: [Control]
# 2. frase:  tengo la cabeza a punto de estallar  ---> Resultado: [Indicador]
# 2. frase:  me muero de sueño  ---> Resultado: [Indicador]
# 2. frase:  me agobio al coger aire  ---> Resultado: [Indicador]
# 1. frase:  me asfixio  ---> Resultado: [Control]
# 2. frase:  siento debilidad en las piernas  ---> Resultado: [Indicador]
# 2. frase:  se me duermen las manos  ---> Resultado: [Control]
