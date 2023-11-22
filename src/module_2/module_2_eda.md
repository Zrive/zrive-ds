Initializing


```python
#imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#retrieving data
path_data = '/Users/alvarochapela/Documents/DATOS_ZRIVE/Module2Data'

orders = pd.read_parquet(f"{path_data}/orders.parquet")
users = pd.read_parquet(f"{path_data}/users.parquet")
regulars = pd.read_parquet(f"{path_data}/regulars.parquet")
inventory = pd.read_parquet(f"{path_data}/inventory.parquet")
abandoned_carts = pd.read_parquet(f"{path_data}/abandoned_carts.parquet")

```

Starting the analysis


```python
n_regulars = regulars.groupby('user_id')['variant_id'].nunique().reset_index().rename(columns={'variant_id':'n_regulars'})
users = users.merge(n_regulars, on='user_id', how = 'left').fillna({'n_regulars':0})
# Veo los users y quito nulls
users.dropna().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>n_regulars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>09d70e0b0778117aec5550c08032d56f8e06f992741680...</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2021-06-28 12:07:04</td>
      <td>2021-06-01 00:00:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4f5ff38ce5ed48096ba80dff80e167db1ad24b9ebdb00c...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2020-06-12 12:07:35</td>
      <td>2020-06-01 00:00:00</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7b2ae50bb11646436fa613394fc3e71e1a0cdc3ba30cdb...</td>
      <td>Proposition</td>
      <td>UKF</td>
      <td>2020-10-03 09:53:57</td>
      <td>2020-10-01 00:00:00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>5e977a4aa2c57f306b8a22f92eaaa177f7dc31a52df82c...</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2021-10-14 10:41:13</td>
      <td>2021-10-01 00:00:00</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>eafb89ad33eb377adb98a915b6a5a65f1284c2db517d07...</td>
      <td>Proposition</td>
      <td>UKH</td>
      <td>2022-01-20 15:53:09</td>
      <td>2022-01-01 00:00:00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Identificar tipos de compradores


```python
user_order_counts = orders.groupby('user_id').size()
frequent_users_ids = user_order_counts[user_order_counts > 1].index
frequent_users_details = users[users['user_id'].isin(frequent_users_ids)]

regular_shoppers = regulars['user_id']
cart_abandoners = abandoned_carts['user_id']
popular_products = orders['id'].explode().value_counts()

loyal_customers = regular_shoppers[regular_shoppers.isin(popular_products.index)]
cart_abandoners_not_regular = cart_abandoners[~cart_abandoners.isin(regular_shoppers)]
```


```python
frequent_users_details.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>n_regulars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>133.000000</td>
      <td>133.000000</td>
      <td>133.000000</td>
      <td>133.000000</td>
      <td>133.000000</td>
      <td>1411.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.977444</td>
      <td>2.052632</td>
      <td>0.834586</td>
      <td>0.090226</td>
      <td>0.669173</td>
      <td>5.343019</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.334088</td>
      <td>0.907175</td>
      <td>0.986119</td>
      <td>0.312823</td>
      <td>0.974849</td>
      <td>11.522472</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>110.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_users_details.hist(bins=100, log=True)
```




    array([[<Axes: title={'center': 'count_people'}>,
            <Axes: title={'center': 'count_adults'}>],
           [<Axes: title={'center': 'count_children'}>,
            <Axes: title={'center': 'count_babies'}>],
           [<Axes: title={'center': 'count_pets'}>,
            <Axes: title={'center': 'n_regulars'}>]], dtype=object)




    
![png](module_2_eda_files/module_2_eda_7_1.png)
    



```python
users['n_regulars'].hist(bins=100, log=True)
```




    <Axes: >




    
![png](module_2_eda_files/module_2_eda_8_1.png)
    



```python
(users['n_regulars']>7).sum()/len(users)
```




    0.09271523178807947




```python
len(frequent_users_details)/len(users)
```




    0.2831627533614289




```python
family_cols = [col for col in users.columns if col.startswith('count_')]
family_inputs = users.count_people.dropna().count()

d = {f"any_{col}": (users[col] > 0).sum()/family_inputs for col in family_cols}
d
```




    {'any_count_people': 0.9938461538461538,
     'any_count_adults': 0.9907692307692307,
     'any_count_children': 0.4,
     'any_count_babies': 0.07076923076923076,
     'any_count_pets': 0.40615384615384614}




```python
#Casas con niños
(users[['count_children', 'count_babies']].sum(axis=1) > 0).sum() / family_inputs
```




    0.4430769230769231



2. REGULAR PRODUCTS



```python
regulars_df = regulars.merge(inventory, on='variant_id', how='left')
regulars_df.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3.49</td>
      <td>3.5</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan, window-glass-cleaner]</td>
    </tr>
  </tbody>
</table>
</div>




```python
clean_regular = regulars_df.dropna()
clean_regular.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 15034 entries, 4 to 18104
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   user_id           15034 non-null  object        
     1   variant_id        15034 non-null  int64         
     2   created_at        15034 non-null  datetime64[us]
     3   price             15034 non-null  float64       
     4   compare_at_price  15034 non-null  float64       
     5   vendor            15034 non-null  object        
     6   product_type      15034 non-null  object        
     7   tags              15034 non-null  object        
    dtypes: datetime64[us](1), float64(2), int64(1), object(4)
    memory usage: 1.0+ MB



```python
clean_regular.groupby(['product_type'])['user_id'].nunique().sort_values(ascending=False).head(40).plot(kind='bar', figsize=(15,5))
```




    <Axes: xlabel='product_type'>




    
![png](module_2_eda_files/module_2_eda_16_1.png)
    


Aquí tenemos las categorías de productos más guardadas en regulars

Ahora vamos a ver la distribución de precios de productos guardados regularmente


```python
# Histograma para observar cómo se distribuyen los precios de los productos guardados regularmente.
plt.figure(figsize=(10, 6))
plt.hist(clean_regular['price'], bins=30, color='blue', edgecolor='black')
plt.title('Distribución de Precios de Productos Guardados Regularmente')
plt.xlabel('Precio')
plt.ylabel('Número de Productos')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_19_0.png)
    


Vamos a comparar los precios para ver si los usuarios tienden a guardar productos en oferta


```python
clean_regular.loc[:, 'is_on_offer'] = clean_regular['price'] < clean_regular['compare_at_price']

# Visualización de cuántos productos están en oferta vs. cuántos no lo están
on_offer_counts = clean_regular['is_on_offer'].value_counts()
plt.figure(figsize=(8, 6))
on_offer_counts.plot(kind='bar', color=['red', 'green'], log=True)
plt.title('Productos En Oferta vs No En Oferta')
plt.xlabel('Está en oferta')
plt.ylabel('Número de Productos')
plt.xticks(rotation=0)
plt.show()
```

    /var/folders/pw/b4ryzt9s5c5g1n_snb_j6yxw0000gn/T/ipykernel_35904/214502736.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      clean_regular.loc[:, 'is_on_offer'] = clean_regular['price'] < clean_regular['compare_at_price']



    
![png](module_2_eda_files/module_2_eda_21_1.png)
    


Los usuarios guardan muchos más productos en oferta

Vamos a ver las tendencias a lo largo del tiempo



```python
monthly_counts = clean_regular.resample('M', on='created_at').size()

```


```python
# Visualizar
monthly_counts.plot()
plt.title("Productos Guardados por Mes")
plt.ylabel("Cantidad")
plt.xlabel("Mes")
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_25_0.png)
    


Vamos a ver los días mas populares


```python
clean_regular['weekday'] = clean_regular['created_at'].dt.day_name()
weekday_counts = clean_regular['weekday'].value_counts()
```

    /var/folders/pw/b4ryzt9s5c5g1n_snb_j6yxw0000gn/T/ipykernel_35904/3488275417.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      clean_regular['weekday'] = clean_regular['created_at'].dt.day_name()



```python
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = weekday_counts.reindex(order)

# Visualizar
weekday_counts.plot(kind='bar')
plt.title("Productos Guardados por Día de la Semana")
plt.ylabel("Cantidad")
plt.xlabel("Día de la Semana")
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_28_0.png)
    


Muy consistente a lo largo de la semana, no hay un día claramente preferido

Vamos a ver los proveedores más populares


```python
vendor_counts = clean_regular['vendor'].value_counts().head(10) 
# Visualizar
vendor_counts.plot(kind='bar')
plt.title("Top 10 Proveedores con Más Productos Guardados")
plt.ylabel("Cantidad")
plt.xlabel("Proveedor")
plt.xticks(rotation=45)
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_31_0.png)
    



```python
orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB


Analisis temporal de orders


```python
#Como se distribuyen las orders por mes
orders.set_index('created_at').resample('M').size().plot(title='Órdenes por Mes')
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_34_0.png)
    



```python
#Tiempo desde que se crea una orden hasta que se hace
orders['time_difference'] = (orders['created_at'] - orders['order_date']).dt.days
orders['time_difference'].plot.hist(title='Distribución del Tiempo entre Creación y Fecha de Orden')
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_35_0.png)
    



```python
#Dias de la semana mas populares
orders['weekday'] = orders['created_at'].dt.dayofweek
orders.groupby('weekday').size().plot(kind='bar', title='Pedidos por Día de la Semana')
plt.show()

```


    
![png](module_2_eda_files/module_2_eda_36_0.png)
    



```python
#Productos mas pedidos por los usuarios
orders.explode('ordered_items').groupby('ordered_items').size().sort_values(ascending=False).head(20).plot(kind='bar', title='Top 20 Ítems Más Pedidos')
plt.show()

```


    
![png](module_2_eda_files/module_2_eda_37_0.png)
    



```python
#Cantidad de items que suelen tener los pedidos
orders['num_items'] = orders['ordered_items'].apply(len)
orders['num_items'].plot.hist(title='Distribución de la Cantidad de Ítems por Pedido')
plt.show()

```


    
![png](module_2_eda_files/module_2_eda_38_0.png)
    

