#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv("./datasets/car_data.csv")
x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,-1].values
print(dataset)

#plot the dataset
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
x_set,y_set = x,y
x1,x2= np.meshgrid(
    np.arange(start= x_set[:,0].min() - 10,
              stop=x_set[:,0].max()+10,
              step=0.25),

    np.arange(start= x_set[:,1].min() - 1000,
              stop=x_set[:,1].max()+1000,
              step=0.25),
)

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],
                x_set[y_set==j,1],
                c=ListedColormap(('salmon','black'))(i),
                label=j)

plt.title("previous campaign data")
plt.xlabel('Age')
plt.ylabel('Extimated Salary')
plt.legend()
plt.show()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_sc = sc.fit_transform(x)

#training the logistic regression model on the previous campaign
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_sc,y)

#visualizing the model result
from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_sc),y

x1,x2= np.meshgrid(
    np.arange(start= x_set[:,0].min() - 10,
              stop=x_set[:,0].max()+10,
              step=0.25),

    np.arange(start= x_set[:,1].min() - 1000,
              stop=x_set[:,1].max()+1000,
              step=0.25),
)

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

plt.contourf(
    x1,
    x2,classifier.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
    alpha=0.75,cmap=ListedColormap(('salmon','blue')))

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],
                x_set[y_set==j,1],
                c=ListedColormap(('salmon','black'))(i),
                label=j)

plt.title("logistic regression - previous campaign data")
plt.xlabel('Age')
plt.ylabel('Extimated Salary')
plt.legend()
plt.show()

