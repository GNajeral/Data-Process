import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
from sklearn.impute import SimpleImputer
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

df = pd.DataFrame(pd.read_excel("output_def.xlsx"))

print(df)

print(df['n_tumor'].shape[0])




# Labelizer contiene el numero de datos que contiene ambas etiquetas (Se ve en crosstable)
def categorical_relation(var1,var2,labelizer):
  crosstable=pd.crosstab(df[var1],df[var2])
  print(crosstable)
  labelizer= labelizer
  mosaic(df,[var1,var2],gap=0.01, title='Relation between ' + var1 + ' and ' + var2,horizontal = False,labelizer = labelizer)

categorical_relation('neoadjuvant','grade',
                      lambda k:{('0','1'):44  ,('0','2'):136  ,('0','3'):18,('1','1'):2,('1','2'):34,('1','3'):13}[k])


#print(type(test['ehr'].count()))
#test['ehr'].count().plot(kind = 'scatter')


##df_preprocessed_def['ki67'] = df_preprocessed_def['ki67'].astype("float")

##df_preprocessed_def = df_preprocessed_def.reset_index(drop=True)
##kia = df_preprocessed_def['ki67'].reset_index(drop=True)
##print(type(kia))