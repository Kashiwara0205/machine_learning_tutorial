from housing_data import HousingData
import matplotlib.pyplot as plt
import seaborn as sns


data = HousingData()
df = data.df

sns.set(style="whitegrid", context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()