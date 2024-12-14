## EXNO-3-DS

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
from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/'Encoding Data.csv'

import pandas as pd
import numpy as np

df=pd.read_csv('drive/MyDrive/Encoding Data.csv')

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

![Screenshot 2024-12-14 084620](https://github.com/user-attachments/assets/20d99fbc-c71f-4881-8e23-1502a8f6810a)

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

![Screenshot 2024-12-14 084627](https://github.com/user-attachments/assets/1cbbd0ba-c1e4-41ab-95a4-3255fb848f10)

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

![Screenshot 2024-12-14 084636](https://github.com/user-attachments/assets/548cfa3b-70ea-4750-b7a9-45cdd6e83d69)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()


enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2

![Screenshot 2024-12-14 084646](https://github.com/user-attachments/assets/34ab3ec1-0b85-425a-a550-75741998508e)

pd.get_dummies(df2,columns=["nom_0"])

![Screenshot 2024-12-14 084653](https://github.com/user-attachments/assets/aa31a754-be92-4654-8dad-d66c91ec8faf)

ls drive/MyDrive/data.csv

![Screenshot 2024-12-14 084721](https://github.com/user-attachments/assets/ba7a7bf9-7407-42ee-8bd9-88db0b384507)

df=pd.read_csv('drive/MyDrive/data.csv')

df

![Screenshot 2024-12-14 084732](https://github.com/user-attachments/assets/7b02dc82-12a9-4c2e-96e5-b133282f8792)

pip install --upgrade category_encoders

![Screenshot 2024-12-14 084748](https://github.com/user-attachments/assets/98561d74-c88c-40ef-8b72-55c58bda65c1)


from category_encoders import BinaryEncoder
df=pd.read_csv('drive/MyDrive/data.csv')

df


![Screenshot 2024-12-14 084804](https://github.com/user-attachments/assets/a0577d1e-36ee-4f44-9c79-949f9b0680a9)


from category_encoders import BinaryEncoder

print(df.columns)

![Screenshot 2024-12-14 084816](https://github.com/user-attachments/assets/a807246a-6e3f-476e-82c9-0de299535b5d)

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb

![Screenshot 2024-12-14 084827](https://github.com/user-attachments/assets/5556ceef-0aca-49d7-bedf-ba0f40f80450)

from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)

cc

![Screenshot 2024-12-14 084837](https://github.com/user-attachments/assets/bd886a74-fa5a-4b0f-9161-e8d7caf2fa16)

from scipy import stats

df=pd.read_csv('drive/MyDrive/Data_to_Transform.csv')

df

![Screenshot 2024-12-14 084844](https://github.com/user-attachments/assets/03521db3-0765-400c-b7dc-c267ec5c2331)

df.skew()

![Screenshot 2024-12-14 084852](https://github.com/user-attachments/assets/53235882-e694-4d07-be21-e26df6b4ac92)

np.log(df["Highly Positive Skew"])

![Screenshot 2024-12-14 084857](https://github.com/user-attachments/assets/2ac28102-2056-4bcb-8fbb-fe58bc556853)

np.reciprocal(df["Moderate Positive Skew"])

![Screenshot 2024-12-14 084904](https://github.com/user-attachments/assets/2d4e6035-1cb4-43be-a7f2-dc1891fbc021)

np.sqrt(df["Highly Positive Skew"])

![Screenshot 2024-12-14 084913](https://github.com/user-attachments/assets/c84081f1-deb9-40d2-9147-a00d77039740)

np.square(df["Highly Positive Skew"])

![Screenshot 2024-12-14 084920](https://github.com/user-attachments/assets/365f8504-14a9-4cb4-bfbd-35e294578b19)

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df

![Screenshot 2024-12-14 084929](https://github.com/user-attachments/assets/f0352390-0174-44d4-9808-e44cfb81fbd5)

df["Moderate Negative Skew_boxcox"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])

df

![Screenshot 2024-12-14 084955](https://github.com/user-attachments/assets/84dd9f1b-60bf-4a55-b0fa-63136b96c1b6)

df.skew()

![Screenshot 2024-12-14 085004](https://github.com/user-attachments/assets/aec851f6-ab90-4e08-8f6e-fbd2b423dcab)

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df

![Screenshot 2024-12-14 085024](https://github.com/user-attachments/assets/a146a082-ce9a-468e-8525-ca20a325e507)

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![Screenshot 2024-12-14 085036](https://github.com/user-attachments/assets/17f3a16f-994f-4492-85ac-16b01579cabc)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

![Screenshot 2024-12-14 085051](https://github.com/user-attachments/assets/b1de014b-0014-4c08-b61c-0b7b18185685)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer (output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![Screenshot 2024-12-14 085059](https://github.com/user-attachments/assets/866ec1a0-c1f6-4902-a0c4-e994553eb01e)
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

![Screenshot 2024-12-14 085108](https://github.com/user-attachments/assets/8a2554e8-f318-4f54-a416-544a709009c2)

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

![Screenshot 2024-12-14 085118](https://github.com/user-attachments/assets/154e0f6e-a230-48e6-9386-8e621dc020a3)

# RESULT:
       The code is run sucessfully.

       
