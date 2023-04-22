# Ex-06-Feature-Transformation\

# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file

# Program:
~~~
Name : kamali.E
Register Number : 212222110015
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("/content/Data_to_Transform.csv")
df
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
~~~

# Output:
![Screenshot 2023-04-21 114930](https://user-images.githubusercontent.com/120567837/233759699-3ac1e8f8-8bb6-48f2-b008-89f7278a42f7.png)

![Screenshot 2023-04-22 085424](https://user-images.githubusercontent.com/120567837/233759752-26930bf5-5ed9-406f-a945-cdcc223eed40.png)













