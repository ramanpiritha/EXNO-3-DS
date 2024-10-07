## EXNO-3-DS
## Name: Piritharaman R
## Reg no: 212223230148

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/3c992dd1-8e33-49a8-a4aa-3b6650d53027)

# ORDINAL ENCODER:
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/0fc334a2-6c22-4d28-ada8-1b1a06572ef6)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/b34aab01-85c8-4370-8ed9-8df0e3a8accf)
# LABEL ENCODER:
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![image](https://github.com/user-attachments/assets/e6410101-e7ad-4289-9faa-51fc10ec3eb5)
```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/87856302-72e6-4bc2-adde-cc083b2f7d41)
# ONEHOT ENCODER:
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/33e9164a-d90e-4cb6-94aa-3a7ca24dfede)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/0de46948-e4f2-4c43-8cde-267a7afb780b)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/4b607190-92a2-4621-9ce1-9a5a6c5229d9)
# BinaryEncoder:
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/dd0c0d25-3438-4dfb-9232-85b7307f991d)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/f3b6cd47-e04f-4e65-9131-4cd04619797b)
# TARGET ENCODER:
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/b39904d1-c778-4269-9385-793b05468be5)
# FEATURE ENGINEERING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/d4e7a011-d948-4161-837a-2ee27f063099)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/933a7361-bf74-4e2e-9dc0-9a8657645dd9)
```

df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df

```
![image](https://github.com/user-attachments/assets/8e89ee88-1064-4a9c-b0ce-084a5085a9e2)
```

df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/5420a007-5913-49ff-a0b6-0004665b1b9c)
```

df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/72fb8c53-8cc3-45b7-be97-7ef0f1be0d0e)
```

df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/a7f73fe6-47b3-4574-a3fa-07a9189a37a5)
# POWER TRANSFORMATION:
```

df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/885f6589-6203-4f72-a795-d0b37a73c52f)
```

df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/7482a29e-4bf6-4e41-b34a-18443f1bd8f8)
```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f56a8703-63eb-4e99-b224-17cff7b0816b)
```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c22363af-891c-4356-8715-8a8d1dd759f0)
```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c0c18594-dfd8-4c21-955c-6eeda9b15827)

# RESULT:
      Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.

       
