import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
from sklearn.impute import SimpleImputer
import math
import numpy as np
import matplotlib.pyplot as plt

df_preprocessed_def = pd.DataFrame(pd.read_excel("output_def.xlsx"))

print(df_preprocessed_def)

print(df_preprocessed_def['n_tumor'].shape[0])


"""
# Bar plot of sides of the tumours
arr = [[df_preprocessed_def['side_left'].astype(int).sum(),df_preprocessed_def['side_right'].astype(int).sum(),df_preprocessed_def['side_unknown'].astype(int).sum()]]

# converting the array to DF
df = pd.DataFrame(arr,
                   index = ['ehr'],
                   columns = ['side_left', 'side_right', 'side_unkwnown'])

print(df)
df.plot.bar()


# Bar plot of the grades of the tumours
arr = [[df_preprocessed_def['grade_1.0'].astype(int).sum(),df_preprocessed_def['grade_2.0'].astype(int).sum(),df_preprocessed_def['grade_3.0'].astype(int).sum()]]

# converting the array to DF
df = pd.DataFrame(arr,
                   index = ['ehr'],
                   columns = ['grade_1.0', 'grade_2.0', 'grade_3.0'])

print(df)
df.plot.bar()



# Bar plot of the hist_type of the tumours
arr = [[df_preprocessed_def['hist_type_ductal'].astype(int).sum(),df_preprocessed_def['hist_type_lobular'].astype(int).sum(),df_preprocessed_def['hist_type_unknown'].astype(int).sum()]]

# converting the array to DF
df = pd.DataFrame(arr,
                   index = ['ehr'],
                   columns = ['hist_type_ductal', 'hist_type_lobular', 'hist_type_unknown'])

print(df)
df.plot.bar()



# Bar plot of the invasive tumours
arr = [[df_preprocessed_def['invasive'].astype(int).sum(), (df_preprocessed_def['invasive'].shape[0] - df_preprocessed_def['invasive'].astype(int).sum())]]

# converting the array to DF
df = pd.DataFrame(arr,
                   index = ['invasive'],
                   columns = ['YES', 'NO'])

print(df)
df.plot.bar()


# Bar plot of the er_positive, pr_positive
arr = [[df_preprocessed_def['er_positive'].astype(int).sum(), df_preprocessed_def['pr_positive'].astype(int).sum()]]

# converting the array to DF
df = pd.DataFrame(arr,
                   index = ['ehr'],
                   columns = ['er_positive', 'pr_positive'])

print(df)
df.plot.bar()


# Scatter plot of the ki67
df_preprocessed_def['ehr'] = df_preprocessed_def.index
print(df_preprocessed_def.loc[:, ['ki67', 'ehr']])

#df_preprocessed_def.loc[:, ['ki67']].plot()

df_preprocessed_def.loc[:, ['ki67', 'ehr']].plot(kind='scatter',    # kind of plot to show
        x='ehr',
        y='ki67'
       )

# Box plot of the ki67
df_preprocessed_def['ehr'] = df_preprocessed_def.index
print(df_preprocessed_def.loc[:, ['ki67', 'ehr']])

#df_preprocessed_def.loc[:, ['ki67']].plot()

df_preprocessed_def.loc[:, ['ki67', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y='ki67'
       )


# Box plot of the ki67
df_preprocessed_def['ehr'] = df_preprocessed_def.index
print(df_preprocessed_def.loc[:, ['ki67', 'ehr']])

df_preprocessed_def.loc[:, ['ki67', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y='ki67',
        title='ki67'
       )

# Box plot of the menarche_age, menopause_age
print(df_preprocessed_def.loc[:, ['menopause_age', 'menarche_age', 'ehr']])

df_preprocessed_def.loc[:, ['menopause_age', 'menarche_age', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y=['menarche_age', 'menopause_age'],
        title='menopause_age'
       )

# Pregnancy analysis
df_preprocessed_def.loc[:, ['pregnancy', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y='pregnancy',
        title='pregnancy'
       )

# Abort analysis
df_preprocessed_def.loc[:, ['abort', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y='abort',
        title='abort'
       )

# Birth analysis
df_preprocessed_def.loc[:, ['birth', 'ehr']].plot(kind='box',    # kind of plot to show
        x='ehr',
        y='birth',
        title='birth'
       )

df = pd.DataFrame(df_preprocessed_def.groupby(['birth'])['ehr'].count())
print(df)

df.plot(kind = 'barh',
         title = 'Birth Horizontal Bar Analysis')

# Caesarean analysis
print(df_preprocessed_def.groupby(['caesarean'])['ehr'].count())
pd.DataFrame(df_preprocessed_def.groupby(['caesarean'])['ehr'].count()).plot(kind = 'barh',
                                                                              title = 'Caesarean Horizontal Bar Analysis')

pd.DataFrame(df_preprocessed_def.groupby(['caesarean'])['ehr'].count()).plot(kind = 'box',
                                                                              title = 'Caesarean Box Analysis')


# 'age_diagnosed' analysis
print(df_preprocessed_def.groupby(['age_diagnosed'])['ehr'].count())
pd.DataFrame(df_preprocessed_def.groupby(['age_diagnosed'])['ehr'].count()).plot(kind = 'barh',
                                                                              title = 'age_diagnosed Horizontal Bar Analysis')

pd.DataFrame(df_preprocessed_def['age_diagnosed']).plot(kind = 'box',
                                                         title = 'age_diagnosed Box Analysis')


age_diagnosed_df = pd.DataFrame(df_preprocessed_def.groupby(['age_diagnosed'])['ehr'].count())
"""

# n_tumor analysis
print(df_preprocessed_def.groupby(['n_tumor'])['ehr'].count())
pd.DataFrame(df_preprocessed_def.groupby(['n_tumor'])['ehr'].count()).plot(kind = 'barh',
                                                                              title = 'n_tumor Horizontal Bar Analysis')

pd.DataFrame(df_preprocessed_def.groupby(['n_tumor'])['ehr'].count()).plot(kind = 'box',
                                                                              title = 'n_tumor Box Analysis')


#print(type(test['ehr'].count()))
#test['ehr'].count().plot(kind = 'scatter')


##df_preprocessed_def['ki67'] = df_preprocessed_def['ki67'].astype("float")

##df_preprocessed_def = df_preprocessed_def.reset_index(drop=True)
##kia = df_preprocessed_def['ki67'].reset_index(drop=True)
##print(type(kia))