import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


def GetData():

    sqlEngine = create_engine('mysql+pymysql://root:*****', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    Data = pd.read_sql("SELECT * FROM ag002.`breast-cancer`", dbConnection);
    pd.set_option('display.expand_frame_repr', False)
    dbConnection.close()


    a = (Data.drop(columns=['id'])).copy()
    a = (a.drop(columns=['class'])).copy()
    data_x = np.array(a.to_numpy()).reshape(-1,9)


    a2 = Data['class'].copy()
    a2.to_frame()
    data_y = np.array(a2.to_numpy())




    x_treino, x_teste, y_treino, y_teste = train_test_split(data_x, data_y, test_size=0.20)
    #print(f"Quantidade de dados para treino: {len(x_treino)}")
    #print(f"Quantidade de dados para teste(avaliação): {len(x_teste)}")

    return x_treino,x_teste,y_treino,y_teste











