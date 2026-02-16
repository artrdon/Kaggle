import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure


plt.style.use('ggplot')
#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# чтение данных
df = pd.read_csv('data/test.csv')
df = df[['Pclass', 'Sex', 'SibSp', 'Parch',
           'Ticket', 'Fare', 'Embarked']]

df.loc[df['Sex'] == 'male', 'Sex'] = '0'
df.loc[df['Sex'] == 'female', 'Sex'] = '1'
df['Sex'] = df['Sex'].astype('int64')
df['Fare'] = df['Fare'].astype('int64')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# отбор числовых колонок
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# отбор нечисловых колонок
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

print("Missing data")
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
