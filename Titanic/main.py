import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

os.system('cls')
print('data collecting')
train = pd.read_csv('data/train.csv', sep=',')
test = pd.read_csv('data/test.csv', sep=',')

# input data
X = train[['Pclass', 'Sex', 'SibSp', 'Parch']]

X.loc[X['Sex'] == 'male', 'Sex'] = '0'
X.loc[X['Sex'] == 'female', 'Sex'] = '1'
X['Sex'] = X['Sex'].astype('int64')


XTEST = test[['Pclass', 'Sex', 'SibSp', 'Parch']]

XTEST.loc[XTEST['Sex'] == 'male', 'Sex'] = '0'
XTEST.loc[XTEST['Sex'] == 'female', 'Sex'] = '1'
XTEST['Sex'] = XTEST['Sex'].astype('int64')
# output data
Y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

os.system('cls')
print('training...')
model = GradientBoostingClassifier(
    learning_rate=0.01,
    n_estimators=1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    warm_start=True
)

train_errors = []
val_errors = []

for i in range(1, 113):
    model.n_estimators = i
    model.fit(X_train, y_train)
    train_err = mean_squared_error(y_train, model.predict(X_train))
    val_err = mean_squared_error(y_val, model.predict(X_val))
    train_errors.append(train_err)
    val_errors.append(val_err)
    if i % 10 == 0:
        print(f'{i} trees')
        print(f'train error: {train_err}')
        print(f'val error:   {val_err}')

os.system('cls')
print("model is trained")
pred = model.predict(XTEST)
print(pred)

plt.figure(figsize=(10, 6))
plt.plot(list(train_errors), label='Train Loss')
plt.plot(list(val_errors), label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

data = {'PassengerId': test['PassengerId'], 'Survived': pred}
df = pd.DataFrame(data)
df.to_csv('people113.csv', index=False)