from preprocessing.preprocessing import Preprocessing
import pandas as pd

def test_preprocessing():
    path_to_data = 'C:/Users/Admin/Desktop/Data-Mining-assignment-1/spaceship-titanic/train.csv'
    preprocessor = Preprocessing(path=path_to_data, target='Transported')
    preprocessed_data = preprocessor.preprocess()
    #print(pd.read_csv(path_to_data))
    #print(preprocessed_data)
    assert(isinstance(preprocessed_data, pd.DataFrame))
