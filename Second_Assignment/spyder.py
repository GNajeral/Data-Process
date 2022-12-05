import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
from sklearn.impute import SimpleImputer
import math
import numpy as np

# Auxiliary functions

def diagnoseDate_to_ageDiagnosed(birthDate, diagnoseDate):
    return pd.to_datetime(diagnoseDate).year-pd.to_datetime(birthDate).year

def deathDate_to_survivalTime(diagnosisDate, deathDate):
    survivalTime = pd.to_datetime(deathDate).year-pd.to_datetime(diagnosisDate).year
    if math.isnan(survivalTime) or survivalTime < 0:
        survivalTime = 1000
    return survivalTime

def deathDate_to_survived(deathDate):
    survived = pd.to_datetime(deathDate).year
    if math.isnan(survived):
        survived = 1
    else: 
        survived = 0
    return survived

def recurrence_year_to_recurrence_time(diagnosisDate, recurrence_year):
    recurrenceTime = recurrence_year-pd.to_datetime(diagnosisDate).year
    if math.isnan(recurrenceTime) or recurrenceTime < 0:
        recurrenceTime = 1000
    return recurrenceTime

def fix_pregnancy(pregnancies, abortions, births):
    if pregnancies < (births + abortions) or pregnancies > (births + abortions):
        pregnancies = births + abortions
    return pregnancies

def preprocess_t(x):
    new_t = x.t
    if pd.isnull(new_t):
        if not(pd.isnull(x.t_after_neoadj)):
            new_t = x.t_after_neoadj
        else:
            new_t = "unknown"
    return new_t

def preprocess_n(x):
    new_n = x.n
    if pd.isnull(new_n):
        if not(pd.isnull(x.n_after_neoadj)):
            new_n = x.n_after_neoadj
        else:
            new_n = "unknown"
    return new_n

def preprocess_m(x):
    new_m = x.m
    if pd.isnull(new_m):
        if not(pd.isnull(x.m_after_neoadj)):
            new_m = x.m_after_neoadj
        else:
            new_m = "unknown"
    return new_m

def fill_t_after_neoadj(x):
    new_t = x.t_after_neoadj
    if pd.isnull(new_t):
            new_t = x.t
    return new_t

def fill_n_after_neoadj(x):
    new_n = x.n_after_neoadj
    if pd.isnull(new_n):
            new_n = x.n
    return new_n

def fill_m_after_neoadj(x):
    new_m = x.m_after_neoadj
    if pd.isnull(new_m):
            new_m = x.m
    return new_m

# Deleting duplicated data and unused column
df1 = pd.read_excel("breast_cancer_data.xlsx")
df1 = df1.drop_duplicates(subset=['ehr'], keep='first')
df1 = df1.set_index('ehr')
df2 = pd.read_excel("breast_cancer_data_2.xlsx")
df2 = df2.drop_duplicates(subset=['ehr'], keep='first')
df2 = df2.set_index('ehr')
df = pd.concat([df1, df2], axis=0)
df.pop('Unnamed: 0')
    
# Duplicating the DataFrame in order to obtain the numerical variables
df_num = pd.DataFrame(data=df, columns=df.columns, index=df.index)
df_num.pop('side')
df_num.pop('neoadjuvant')
df_num.pop('grade')
df_num.pop('invasive')
df_num.pop('er_positive')
df_num.pop('pr_positive')
df_num.pop('her2_positive')
df_num.pop('hist_type')

# Dividing the DataFrame into categorical and numerical variables
num_cols = df_num.columns.tolist()
df_cat = df.drop(num_cols, axis=1)

df_cat.side = df_cat.side.apply(lambda x: 'unknown' if (x != 'left' and x != 'right') else x)
df_cat.invasive = df_cat.invasive.apply(lambda x: 0 if x != 1 else x)

# Imputation of nulls in categorical columns using Simple Imputer
imp_cat = SimpleImputer(strategy='most_frequent')
columns = df_cat.columns
index = df_cat.index
df_cat = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=columns, index=index)

# Taking some variables out as they are already converted into numerical values
df_aux = pd.DataFrame(data=df_cat, columns=df_cat.columns, index=df_cat.index)
df_cat.pop('invasive')
df_cat.pop('er_positive')
df_cat.pop('pr_positive')
df_cat.pop('her2_positive')
num_cols = df_cat.columns.tolist()
df_aux = df_aux.drop(num_cols, axis=1)

# Using OneHotEncoder
ohe = preprocessing.OneHotEncoder(sparse=False)
df_cat_ohe = pd.DataFrame(ohe.fit_transform(df_cat), 
                          columns=ohe.get_feature_names_out(df_cat.columns.tolist()),
                          index=df_cat.index)

# Merge both DataFrames (df_cat_ohe and df_aux)
df_cat_def = pd.merge(left=df_cat_ohe, right=df_aux, on='ehr')

# Age at which the patient was diagnosed
ageDiagnosed = pd.Series(df_num.apply(lambda x: diagnoseDate_to_ageDiagnosed(x.birth_date, x.diagnosis_date), axis=1), name='age_diagnosed')

# Time of survival since diagnosis, 1000 in case of full recovery
survivalTime = pd.Series(df_num.apply(lambda x: deathDate_to_survivalTime(x.diagnosis_date, x.death_date), axis=1), name='survival_time')

# We set "survived" column to be the target variable
class_col = pd.Series(df_num.apply(lambda x: deathDate_to_survived(x.death_date), axis=1), name='survived')

# Recurrence time for a patient
recurrenceTime = pd.Series(df_num.apply(lambda x: recurrence_year_to_recurrence_time(x.diagnosis_date, x.recurrence_year), axis=1), name='recurrence_time')

# Changing variables
df_num.pop('birth_date')
df_num.pop('diagnosis_date')
df_num.pop('death_date')
df_num.pop('recurrence_year')
df_num = pd.merge(left=df_num, right=ageDiagnosed, on='ehr')
df_num = pd.merge(left=df_num, right=survivalTime, on='ehr')
df_num = pd.merge(left=df_num, right=recurrenceTime, on='ehr')

df_num.pregnancy = df_num.pregnancy.apply(lambda x: 0 if math.isnan(x) else x)
df_num.abort = df_num.abort.apply(lambda x: 0 if math.isnan(x) else x)
df_num.birth = df_num.birth.apply(lambda x: 0 if x < 0 else x)
df_num.caesarean = df_num.caesarean.apply(lambda x: 0 if math.isnan(x) else x)

df_num.pregnancy = df_num.apply(lambda x: fix_pregnancy(x.pregnancy, x.abort, x.birth), axis=1)

# Imputation of nulls in numerical columns using Simple Imputer
imp_num = SimpleImputer(strategy='mean')
columns = df_num.columns
index = df_num.index
df_num_def = pd.DataFrame(imp_num.fit_transform(df_num), columns=columns, index=index)

# We round up the menarche_age and menopause_age columns to give it sense
df_num_def.menarche_age = df_num_def.menarche_age.apply(np.ceil)
df_num_def.menopause_age = df_num_def.menopause_age.apply(np.ceil)

df_preprocessed = pd.merge(left=df_cat_def, right=df_num_def, on='ehr')
df_preprocessed.to_excel("output1.xlsx")

# Deleting duplicated data
df3 = pd.read_csv("breast_cancer_data_tnm.csv")
df3 = df3.drop_duplicates(subset=['ehr'], keep='last')
df3 = df3.set_index('ehr')
df4 = pd.read_csv("breast_cancer_data_tnm_2.csv")
df4 = df4.drop_duplicates(subset=['ehr'], keep='last')
df4 = df4.set_index('ehr')
df_tnm = pd.concat([df3, df4], axis=0)

# Preprocess of t, n and m columns
df_tnm.t = df_tnm.apply(lambda x: preprocess_t(x), axis=1)
df_tnm.n = df_tnm.apply(lambda x: preprocess_n(x), axis=1)
df_tnm.m = df_tnm.apply(lambda x: preprocess_m(x), axis=1)

# Preprocess of t_after_neoadj, n_after_neoadj and m_after_neoadj columns
df_tnm.t_after_neoadj = df_tnm.apply(lambda x: fill_t_after_neoadj(x), axis=1)
df_tnm.n_after_neoadj = df_tnm.apply(lambda x: fill_n_after_neoadj(x), axis=1)
df_tnm.m_after_neoadj = df_tnm.apply(lambda x: fill_m_after_neoadj(x), axis=1)

df_tnm.t = df_tnm.t.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm.n = df_tnm.n.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm.m = df_tnm.m.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm.t_after_neoadj = df_tnm.t_after_neoadj.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm.n_after_neoadj = df_tnm.n_after_neoadj.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm.m_after_neoadj = df_tnm.m_after_neoadj.apply(lambda x: x.replace('.0','') if isinstance(x, str) else int(x))
df_tnm = df_tnm.astype(str)

# Taking variables out as they do not need to be applied with OHE
df_tnm_aux = pd.DataFrame(data=df_tnm, columns=df_tnm.columns, index=df_tnm.index)
df_tnm.pop('n_tumor')
num_cols = df_tnm.columns.tolist()
df_tnm_aux = df_tnm_aux.drop(num_cols, axis=1)

# Using OneHotEncoder
df_tnm_ohe = pd.DataFrame(ohe.fit_transform(df_tnm), 
                          columns=ohe.get_feature_names_out(df_tnm.columns.tolist()),
                          index=df_tnm.index)

# Merging DataFrames back
df_tnm_def = pd.merge(left=df_tnm_aux, right=df_tnm_ohe, on='ehr')
df_tnm_def.to_excel("output2.xlsx")

df_preprocessed_pre_def = pd.merge(left=df_preprocessed, right=df_tnm_def, how='outer', on='ehr')

# Imputation of the new nulls (caused by merging 2 different datasets) using Simple Imputer
imp_df = SimpleImputer(strategy='most_frequent')
columns = df_preprocessed_pre_def.columns
index = df_preprocessed_pre_def.index
df_preprocessed_def = pd.DataFrame(imp_df.fit_transform(df_preprocessed_pre_def), columns=columns, index=index)
#df_preprocessed_def.to_excel("output_def.xlsx")

print(df_preprocessed_def)