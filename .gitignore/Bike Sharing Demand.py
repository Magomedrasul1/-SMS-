#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy
from sklearn import metrics, linear_model, model_selection, pipeline, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb


# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


datatrain = pd.read_csv('train.csv')
datatest = pd.read_csv('test.csv')


# In[4]:


# Видно, что последние два признака определяют целевую переменную, их неообходимо исключить из выборки
datatrain.head(-10)


# In[5]:


datatest.head(-10)


# In[6]:


# Проверим, есть ли провущенные значения
datatrain.isnull().values.any()


# In[7]:


datatrain.info()


# In[8]:


# Преобразуем данные даты в удобной для нас обьект и создадим два  дополнительных признака - месяц и час
datatrain['datetime'] = datatrain['datetime'].apply(pd.to_datetime)
datatrain['month'] = datatrain['datetime'].apply(lambda x : x.month)
datatrain['hour'] = datatrain['datetime'].apply(lambda x : x.hour)


# In[9]:


# создаем две выборки для обучения и теста. Лучшее разделение с точки зрения поствленной задачи будет
# разделение на прошлое и будущее
datatrain_train = datatrain.iloc[: - 1000, :] # данные для обучения
datatrain_test = datatrain.iloc[-1000:, :] # тест данные
print(datatrain_train.shape, datatrain_test.shape) # размерности обучающей и тестовой выборок


# In[10]:


# Удаляем ненужные  признаки
delerPrizTr = ['casual', 'registered', 'datetime']
delerPrizTt = ['casual', 'registered', 'datetime']
datatrain_train = datatrain_train.drop(delerPrizTr, axis = 1)
datatrain_test = datatrain_test.drop(delerPrizTt, axis = 1)
train_priz = datatrain_train.drop('count', axis = 1)
train_otv = datatrain_train['count'].values
test_priz = datatrain_test.drop('count', axis = 1)
test_otv = datatrain_test['count'].values


# In[11]:


# Создаем маски для последующего нормирования данных
catig = ['season', 'weather', 'month'] 
# Категориальные признаки
maskCatig = np.array([(colum in catig) for colum in train_priz.columns])
print(maskCatig)
binar = ['workingday', 'holiday'] 
# Бинарные признаки
maskBinar = np.array([(colum in binar) for colum in train_priz.columns])
print(maskBinar)
# Числовые признаки
num = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
maskNum = np.array([(colum in num) for colum in train_priz.columns])
print(maskNum)


# In[12]:


# regressor = RandomForestRegressor(random_state = 0,  n_estimators = 50)
# forest_params = {'max_depth': range(1,11), 'max_features': range(4,19)}
# forest_grid = model_selection.GridSearchCV(regressor, forest_params, cv=5)
# forest_grid(train_priz, train_otv)

# regressor = RandomForestRegressor(random_state = 0, max_depth = 30, n_estimators = 50, max_features = 20)
# regressor.fit(train_priz, train_otv)
# regpred = regressor.predict(test_priz)
# metrics.mean_absolute_error(test_otv, regpred)


# In[13]:


# regressor = linear_model.SGDRegressor(random_state = 0)
regressor = RandomForestRegressor(random_state = 0, max_depth = 30, n_estimators = 50, max_features = 20)
# regressor = KNeighborsClassifier(n_neighbors=50)
# Разделим данные и отнормируем
esm = pipeline.Pipeline(steps = [
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [
#         отделяем бинарные данные
        ('binary_variables_processing', preprocessing.FunctionTransformer(lambda df : df[: , maskBinar])),
#         отделяем числовые данные и нормируем 
        ('numeric_variables_processing', pipeline.Pipeline(steps = [
            ('selecting', preprocessing.FunctionTransformer(lambda df : df[: , maskNum])), 
            ('scaling', preprocessing.StandardScaler(with_mean = 0, with_std = 1))])),
#         отделяем категориальные данные и трансформируем
        ('categorical_variables_processing', pipeline.Pipeline(steps = [
            ('selecting', preprocessing.FunctionTransformer(lambda df : df[: , maskCatig])),
            ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))
        ])),
    ])),
#     применяем наш алгоритм
    ('model_fitting', regressor) 
])


# In[14]:


esm.fit(train_priz, train_otv) # обучаем алгоритм


# In[15]:


# проверка работы алгоритма
esmpred = esm.predict(test_priz)
metrics.mean_absolute_error(test_otv, esmpred)

Ошибка при использовании случайного леса достигла 79 великов. При остальных больше 130 великов. Необходимо дополнительно проанализировать данные. Есть вариант поработать с признаками. Попробую завтра!
# In[81]:





# In[ ]:




