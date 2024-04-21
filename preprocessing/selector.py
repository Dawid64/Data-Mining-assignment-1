from abc import ABC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Selector(ABC):
    def select(self, dataset: pd.DataFrame, var_treshold:float) -> pd.DataFrame:
        new_dataset = self.drop_unique(dataset)
        new_dataset = self.drop_low_var(dataset, var_treshold)
        new_dataset = self.drop_high_corr(dataset)
        return new_dataset

    def drop_unique(self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        Drops columns with unique or very rarely repeating values that are not floats
        '''
        new_dataset = dataset.copy()  
        threshold = len(new_dataset) * 0.01

        for column in new_dataset.columns:
            if new_dataset[column].dtype != 'float64':
                freq_most_common = new_dataset[column].value_counts().max()
                
                if freq_most_common < threshold:
                    new_dataset.drop(column, axis=1, inplace=True)
        
        return new_dataset
    
    def drop_low_var(dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
        '''
        Drops columns from the dataset with variances below the given threshold.
        '''
        data_num = dataset.select_dtypes(include=['number'])
        dataset_scaled = MinMaxScaler().fit_transform(data_num.to_numpy())
        scaled_dataframe = pd.DataFrame(
            dataset_scaled, columns=data_num.columns)
        
        variances = scaled_dataframe.var() 
        low_variance_columns = variances[variances < threshold].index

        dataset_filtered = dataset.drop(columns=low_variance_columns)
        return dataset_filtered
    
    def drop_high_corr(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
