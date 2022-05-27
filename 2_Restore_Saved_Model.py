import keras.models
import numpy as np
import os

cwd = os.getcwd()

modelsaving_path = os.path.join(cwd, "SavedModelalterado")


sample_text = [' me hvoy a zampar un bocata de lomo queso y panceta']

model= keras.models.load_model(modelsaving_path)
# model2 = keras.models.load_model(modelsaving_path_meta)
# NORMAL
for i in sample_text:
    # print(i, (model.predict(np.array([i]))))
    if model.predict(np.array([i]))[0][0] >= 0:
        print("frase: ",i, " ---> Resultado: [Control]")
    else:
        print("frase: ", i, " ---> Resultado: [Indicador]")







# # modelo doble
# for i in sample_text:
#     # print(i, (model.predict(np.array([i]))))
#     if model.predict(np.array([i]))[0][0] >= 0:
#         print("1. frase: ",i, " ---> Resultado: [Control]")
#     elif model.predict(np.array([i]))[0][0] <= 0 and model2.predict(np.array([i]))[0][0] >= 0:
#         print("2. frase: ", i, " ---> Resultado: [Control]")
#     elif model.predict(np.array([i]))[0][0] <= 0 and model2.predict(np.array([i]))[0][0] <= 0:
#         print("2. frase: ",i, " ---> Resultado: [Indicador]")
