import os
import pandas as pd

cwd = os.getcwd()

dict_name_control = "cleaned_control - copia.txt"
dict_name_indicator = "cleaned_Estrés.txt"
CSV_FILE_NAME = "Libraries/dic_control-ingredientes.csv"
DICT_PATH = os.path.join(cwd, "Dictionaries")
FILE_PATH_CONTROL = os.path.join(DICT_PATH, dict_name_control)
FILE_PATH_INDICATOR = os.path.join(DICT_PATH, dict_name_indicator)

def showFiles():
    archivos = list(os.listdir(DICT_PATH))
    for i in archivos:
        print(i)


def openControlFile():
    f = open(FILE_PATH_CONTROL, "r", encoding = 'utf-8')
    count = 0
    dic = []
    print("####################################################\nWords:")
    for i in f:
        if str(i[:-1]) == "":
            pass
        else:
            dic.append([int(1), str(i[:-1])])

        # print(i[0:-5],i[-2])
        # if(int(i[-2]) != 0):
        #     count = count + 1
        #     data_list.append([i[0:-5],i[-2]])
        #     alto_riesgo.append(i[0:-5])
    for j in dic:
        if j == "" :
            print("ojoooooooooooooooooooooooo")

    print("nuevo dic_control:", len(dic))
    for j in dic:
        count = count + 1
        print(count, j)

    return count, dic


def openIndicatorFile(dic_2, FILE_PATH_INDICATOR_fn, dict_name_indicator_fn, count_dic_fn):
    f = open(FILE_PATH_INDICATOR_fn, "r", encoding =  'utf-8')# encoding='utf-8-sig')'windows-1252'
    count = 0
    count_indicator = 0
    print("####################################################\nWords:")
    for i in f:
        if str(i[:-1]) == "":
            pass
        else:
            count_indicator = count_indicator + 1
            string = i.replace(u'\xa0', u' ')
            dic_2.append([int(0), str(string[:-2])])

    for j in dic_2:
        count = count + 1
        print(count, j)
    print("#############################\nTrain Data: ", len(dic_2),"\nControl: ", count_dic_fn, "Dic:", count_indicator)
    return dic_2

showFiles()
count_dic, control_dic = openControlFile()
final_dic = openIndicatorFile(control_dic, FILE_PATH_INDICATOR, dict_name_indicator, count_dic)
df_dic = pd.DataFrame(final_dic )
df_dic.to_csv(CSV_FILE_NAME, header=None, index=None,encoding = 'utf-8')























