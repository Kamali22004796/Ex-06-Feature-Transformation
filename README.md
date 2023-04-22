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

![Screenshot 2023-04-22 085531](https://user-images.githubusercontent.com/120567837/233760285-1680e6e1-1959-44d4-9b58-92793604cb53.png)

![Screenshot 2023-04-22 085504](https://user-images.githubusercontent.com/120567837/233760340-a39ec9f9-730f-41e5-9469-255d72614a09.png)

![Screenshot 2023-04-22 085531](https://user-images.githubusercontent.com/120567837/233760444-906f00fc-7dc9-42c9-a755-6f444c5d4ffb.png)

![Screenshot 2023-04-22 092058](https://user-images.githubusercontent.com/120567837/233760837-2abaf290-800b-45b5-b96a-d21786391f27.png)

[Screenshot 2023-04-22 092058](https://user-images.githubusercontent.com/120567837/233760776-5469705c-c822-4e1f-952c-8fe9d7ad0b5e.png)

![Screenshot 2023-04-22 085620](https://user-images.githubusercontent.com/120567837/233760873-6daed830-e749-4c11-ab70-61c9d2c732d5.png)

![Screenshot 2023-04-22 092621](https://user-images.githubusercontent.com/120567837/233761010-6e36c480-bbbe-4538-b231-5a80824d7e81.png)

![Screenshot 2023-04-22 085643](https://user-images.githubusercontent.com/120567837/233760934-76363606-d737-4393-8eef-a869711d86a2.png)

![Screenshot 2023-04-22 092730](https://user-images.githubusercontent.com/120567837/233761081-81f453f1-d548-4be3-b670-9e3e5b0ce9d7.png)

# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully


















