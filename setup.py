from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

#drop by removing or keeping
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action
    
  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: DropColumnsTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    remaining_set = set(self.column_list) - set(X.columns.to_list())
    assert not remaining_set, f'DropColumnsTransformer.transform unknown columns {remaining_set}'
    
    X_ = X.copy()
    if self.action=='drop':
      X_ = X_.drop(columns=self.column_list)
    else:
      X_ = X_[self.column_list]
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
#This class maps values in a column, numeric or categorical.
#Importantly, it does not change NaNs, leaving that for the imputer step.
class MappingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
 
  def fit(self, X, y = None):
    print("Warning: OHETransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'OHETransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'OHETransformer.transform unknown column {self.target_column}'
    X_ = X.copy()
    X_ = pd.get_dummies(X_, columns=[self.target_column],
                        dummy_na=self.dummy_na,
                        drop_first = self.drop_first)
    print(f'OHETransformer columns: {X_.columns.to_list()}')
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column
    
  def fit(self, X, y = None):
    print("Warning: Sigma3Transformer.fit does nothing.")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'Sigma3Transformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    mean = X_[self.target_column].mean()
    sigma = X_[self.target_column].std()
    high_wall = mean + 3*sigma
    low_wall = mean - 3*sigma
    print(f'Sigma3Transformer mean, sigma, low_wall, high_wall: {round(mean, 2)}, {round(sigma, 2)}, {round(low_wall, 2)}, {round(high_wall, 2)}')
    X_[self.target_column] = X_[self.target_column].clip(lower=low_wall, upper=high_wall)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
  
#from week 2
titanic_transformer_v1 = Pipeline(steps=[
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ], verbose=True)
