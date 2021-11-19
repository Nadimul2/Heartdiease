import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart.csv')
pd.set_option('display.max_columns', None)

missing =df.isnull().sum()
df.drop(df[['FastingBS', 'RestingECG', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']],
        axis=1, inplace=True)

sns.set_theme()
sns.countplot(x='Sex', data=df)
plt.title('Sex Comparison')
plt.show()

sns.kdeplot(x='Cholesterol', fill=True, data=df)
plt.title('Cholesterol Density Chart')
plt.show()

sns.kdeplot(x='RestingBP', fill=True, data=df, color='red')
plt.title('Resting BP Density Chart')
plt.show()

sns.countplot(x='ChestPainType', data=df)
plt.title('Chest Pain Type Comparison')
plt.show()

#print(df[df['Cholesterol']<51])

print(df)
le = LabelEncoder()
df['PainType'] = le.fit_transform(df['ChestPainType'])
# ['ATA'=1 'NAP'=2 'ASY'=0 'TA'=3]
df['sex'] = le.fit_transform(df['Sex'])
# M=1, F=0
df.drop(df[['ChestPainType','Sex']],
        axis=1, inplace=True)

x = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=False)
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(x_train, y_train)
# Age, RestingBP, Cholesterol, PainType, Sex, MaxHR
#predict = model.predict([[]])
predict = model.predict(x_test)

# HeartDisease, YES=1 , NO=0
score= accuracy_score(y_test, predict)
#max=0.7826
print(score)
