from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class FormatNaNColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        inputer = SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='mean',  # la estrategia elegida es cambiar el valor faltante por una constante
            #fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        data = X.copy()
        dataFormatted =  pd.DataFrame(inputer.fit_transform(data[self.columns].values), columns=self.columns, index = data.index)
        data[self.columns] = dataFormatted
        return data

class NormalizeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        scaler = MinMaxScaler()
        data = X.copy()
        df_scaled =  pd.DataFrame(scaler.fit_transform(data[self.columns].values), columns=self.columns, index = data.index)
        data[self.columns] = df_scaled
        return data
