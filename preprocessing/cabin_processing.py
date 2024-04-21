from abc import ABC
import pandas as pd

def cabin_to_components(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset

def cabin_to_side(dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        Changes the content of cabin to just its side.
        For example 'A/1/S' -> 'S'
        '''
        
        new_dataset = dataset.copy()
        new_dataset['Cabin'] = new_dataset['Cabin'].apply(lambda x: x[-1] if isinstance(x, str) else x)
        return new_dataset