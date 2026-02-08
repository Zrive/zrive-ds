```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df_abandoned_carts = pd.read_parquet('/home/raulherrero/datos-zrive/abandoned_carts.parquet')
df_inventory = pd.read_parquet('/home/raulherrero/datos-zrive/inventory.parquet')
df_orders = pd.read_parquet('/home/raulherrero/datos-zrive/orders.parquet')
df_regulars = pd.read_parquet('/home/raulherrero/datos-zrive/regulars.parquet')
df_users = pd.read_parquet('/home/raulherrero/datos-zrive/users.parquet')
```


```python
df_abandoned_carts.head()

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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_abandoned_carts['created_at'].describe()
```


    count                          5457
    mean     2021-12-20 11:07:10.198460
    min             2020-05-20 13:53:24
    25%             2021-11-13 19:52:17
    50%             2021-12-27 13:14:57
    75%             2022-01-30 08:35:19
    max             2022-03-13 14:12:10
    Name: created_at, dtype: object



```python
df_inventory.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB



```python
df_inventory.head()
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
      <th>variant_id</th>
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
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_orders.info()
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



```python
df_orders.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_regulars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB



```python
df_regulars.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB



```python
df_users.dropna().head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4751</th>
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
    </tr>
    <tr>
      <th>3154</th>
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
    </tr>
    <tr>
      <th>736</th>
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
    </tr>
    <tr>
      <th>4792</th>
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
    </tr>
    <tr>
      <th>2217</th>
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
    </tr>
  </tbody>
</table>
</div>


# Dataset de carros abandonados
Que tipo de productos se abandonan mas?



```python
df_exploded_abandoned_carts = df_abandoned_carts.explode('variant_id')
df_abandoned_carts_per_product = df_abandoned_carts_per_product.merge(df_inventory, on='variant_id', how='left')

```

La gente devuelve mas algun tipo de productos?


```python
# Agrupar por product_type y sumar el número de devoluciones
devoluciones_por_tipo = df_abandoned_carts_per_product.groupby('product_type')['num_devoluciones'].sum()

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
devoluciones_por_tipo.plot(kind='bar', color='skyblue')
plt.title('Número de devoluciones por tipo de producto')
plt.xlabel('Tipo de producto')
plt.ylabel('Número de devoluciones')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


```python
# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(df_abandoned_carts_per_product['price'], df_abandoned_carts_per_product['num_devoluciones'], color='skyblue', alpha=0.5)
plt.title('Relación entre precio del producto y número de devoluciones')
plt.xlabel('Precio del producto')
plt.ylabel('Número de devoluciones')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

Parece que no existe una relacion fuerte entre el precio y el numero de ser devueltas fuera del carrito

# Dataset inventory
Que tipo de productos vendemos, cuantos productos por categoria existen
Relacionar con orders y ver que productos se venden mas



```python
plt.figure(figsize=(10, 6))
plt.hist(df_inventory['price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precios de Productos')
plt.grid(True)
plt.show()
```


```python
productos_por_tipo = df_inventory['product_type'].value_counts()

plt.figure(figsize=(10, 6))
productos_por_tipo.plot(kind='bar', color='skyblue')

plt.xlabel('Tipo de Producto')
plt.ylabel('Número de Productos')
plt.title('Número de Productos por Tipo de Producto')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

```


```python
df_exploded_orders = df_orders.explode('ordered_items')
df_exploded_orders = df_exploded_orders['ordered_items'].rename('variant_id')
df_exploded_orders = df_exploded_orders.to_frame()
df_orders_inventory= df_exploded_orders.merge(df_inventory, on='variant_id', how='left')
```


```python
df_orders_inventory.dropna().head()
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
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>71</th>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>76</th>
      <td>33667206054020</td>
      <td>17.99</td>
      <td>20.65</td>
      <td>ecover</td>
      <td>delicates-stain-remover</td>
      <td>[cruelty-free, delicates-stain-remover, eco, v...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>33667206283396</td>
      <td>9.99</td>
      <td>12.00</td>
      <td>ecover</td>
      <td>fabric-softener-freshener</td>
      <td>[cruelty-free, eco, fabric-softener-freshener,...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>39459277602948</td>
      <td>5.79</td>
      <td>5.98</td>
      <td>ecloth</td>
      <td>cleaning-products</td>
      <td>[eco, sponges-cloths-gloves]</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize=(10, 6))
plt.hist(df_orders_inventory['price'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precios de Productos')
plt.grid(True)
plt.show()
```


```python
plt.figure(figsize=(10, 6))
df_orders_inventory['product_type'].value_counts().plot(kind='bar', color='skyblue')

plt.xlabel('Tipo de Producto')
plt.ylabel('Número de Productos')
plt.title('Número de Productos por Tipo de Producto')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

Existen tipos de productos que tienen muchos productos y son muy vendidos (cleaning-products, tins-packaged-foods)
Y existen otros tipos de productos que basan sus ventas en menos porductos (long-life-milk-sustitutes)


```python
plt.figure(figsize=(10, 6))
df_orders_inventory['vendor'].value_counts().head(15).plot(kind='bar', color='skyblue')

plt.xlabel('Tipo de Producto')
plt.ylabel('Número de Productos')
plt.title('Número de Productos por Tipo de Producto')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

# Orders df 
Han aumentado con el tiempo las ventas?


```python
df_exploded_orders = df_orders.explode('ordered_items')
df_exploded_orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 107958 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count   Dtype         
    ---  ------          --------------   -----         
     0   id              107958 non-null  int64         
     1   user_id         107958 non-null  object        
     2   created_at      107958 non-null  datetime64[us]
     3   order_date      107958 non-null  datetime64[us]
     4   user_order_seq  107958 non-null  int64         
     5   ordered_items   107958 non-null  object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 5.8+ MB



```python
# Agrupar por fecha y contar el número de ventas en cada fecha
ventas_por_fecha = df_exploded_orders.groupby(df_exploded_orders['order_date'].dt.date).size()

# Crear el gráfico de la serie temporal
plt.figure(figsize=(10, 6))
ventas_por_fecha.plot()
plt.title('Número de Ventas a lo largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Número de Ventas')
plt.grid(True)
plt.show()
```

# Regulars df
Que tipo de productos se marcan como regulares
Que tipo de cliente tienen mas productos regulares


```python
df_regulars_inventory= df_regulars.merge(df_inventory, on='variant_id', how='left')
df_regulars_inventory.dropna().head()
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
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3.49</td>
      <td>3.50</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan, window-glass-cleaner]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667182493828</td>
      <td>2020-05-06 14:42:11</td>
      <td>4.29</td>
      <td>5.40</td>
      <td>bulldog</td>
      <td>skincare</td>
      <td>[cruelty-free, eco, facial-skincare, vegan]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d883991facbc3b07b62da342d00c97d1e6cea8d2176695...</td>
      <td>33667198910596</td>
      <td>2020-07-06 10:12:08</td>
      <td>14.99</td>
      <td>16.55</td>
      <td>ecover</td>
      <td>dishwashing</td>
      <td>[cruelty-free, dishwasher-tablets, eco, vegan]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>66a195720d6988ff4d32155cc03631b84f68b34d3b0a1e...</td>
      <td>33826459320452</td>
      <td>2020-07-06 17:17:52</td>
      <td>5.09</td>
      <td>5.65</td>
      <td>treeoflife</td>
      <td>snacks-confectionery</td>
      <td>[christmas, nuts-dried-fruit-seeds, organic]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0b7e02fee4b9e215da3bdae70050f20c5ffd18264454a5...</td>
      <td>33667247276164</td>
      <td>2020-07-18 16:56:55</td>
      <td>2.49</td>
      <td>3.00</td>
      <td>method</td>
      <td>hand-soap-sanitisers</td>
      <td>[cruelty-free, eco, hand-soap, vegan]</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize=(10, 6))
plt.hist(df_regulars_inventory['price'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precios de Productos')
plt.grid(True)
plt.show()
```


```python
plt.figure(figsize=(10, 6))
df_regulars_inventory['vendor'].value_counts().head(15).plot(kind='bar', color='skyblue')

plt.xlabel('Tipo de Producto')
plt.ylabel('Número de Productos')
plt.title('Número de Productos por Tipo de Producto')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


```python
plt.figure(figsize=(10, 6))
df_regulars_inventory['product_type'].value_counts().head(15).plot(kind='bar', color='skyblue')

plt.xlabel('Tipo de Producto')
plt.ylabel('Número de Productos')
plt.title('Número de Productos por Tipo de Producto')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


```python
df_regulars_users= df_regulars.merge(df_users, on='user_id', how='left')
df_regulars_users.dropna().head()
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
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-01 00:00:00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
      <td>Proposition</td>
      <td>UKH</td>
      <td>2020-05-06 10:23:11</td>
      <td>2020-05-01 00:00:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-01 00:00:00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
      <td>Proposition</td>
      <td>UKH</td>
      <td>2020-05-06 10:23:11</td>
      <td>2020-05-01 00:00:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2020-05-06 16:03:35</td>
      <td>2020-05-01 00:00:00</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Calcula el recuento de productos regulares por categoría de usuario
productos_por_categoria = df_regulars_users['user_segment'].value_counts()

# Calcula el recuento de usuarios por categoría de usuario
usuarios_por_categoria = df_users['user_segment'].value_counts()

# Crea un DataFrame combinado con ambos recuentos
df_combined = pd.concat([productos_por_categoria, usuarios_por_categoria], axis=1)
df_combined.columns = ['Productos Regulares', 'Usuarios']

# Crea el gráfico de barras
plt.figure(figsize=(10, 6))
df_combined.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Número de Productos Regulares y Usuarios por Categoría de Usuario')
plt.xlabel('Categoría de Usuario')
plt.ylabel('Recuento')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='upper right')
plt.show()
```


```python
# Calcula el recuento de productos regulares por categoría de usuario
productos_por_categoria = df_regulars_users['user_nuts1'].value_counts()

# Calcula el recuento de usuarios por categoría de usuario
usuarios_por_categoria = df_users['user_nuts1'].value_counts()

# Crea un DataFrame combinado con ambos recuentos
df_combined = pd.concat([productos_por_categoria, usuarios_por_categoria], axis=1)
df_combined.columns = ['Productos Regulares', 'Usuarios']

# Crea el gráfico de barras
plt.figure(figsize=(10, 6))
df_combined.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Número de Productos Regulares y Usuarios por Categoría de Usuario')
plt.xlabel('Categoría de Usuario')
plt.ylabel('Recuento')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='upper right')
plt.show()
```

# DF users


```python
df_users.info()
df_users['user_nuts1'].unique()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB



    array(['UKH', 'UKJ', 'UKD', 'UKI', 'UKE', 'UKK', 'UKF', 'UKL', 'UKC',
           'UKG', 'UKM', None, 'UKN'], dtype=object)



```python
df_users_sin_na = df_users.dropna(subset=['count_people'])
```


```python
df_users_sin_na.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.787692</td>
      <td>2.003077</td>
      <td>0.707692</td>
      <td>0.076923</td>
      <td>0.636923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365753</td>
      <td>0.869577</td>
      <td>1.026246</td>
      <td>0.289086</td>
      <td>0.995603</td>
    </tr>
    <tr>
      <th>min</th>
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
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_users_y_orders_sin_na = pd.merge(df_orders, df_users_sin_na, on='user_id', how='inner')
```


```python
df_users_y_orders = pd.merge(df_orders, df_users, on='user_id', how='inner')
```


```python
df_users_y_orders.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
      <th>number_items</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
      <td>14</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3804928049284</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2021-08-31 11:37:20</td>
      <td>2021-08-31</td>
      <td>2</td>
      <td>[33667198910596, 33667247341700, 3408158988710...</td>
      <td>11</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
      <td>25</td>
      <td>Proposition</td>
      <td>UKM</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2273377058948</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-05-23 15:14:42</td>
      <td>2020-05-23</td>
      <td>2</td>
      <td>[33667174465668, 33667214966916, 3366730753651...</td>
      <td>24</td>
      <td>Proposition</td>
      <td>UKM</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3638425026692</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2021-02-24 22:16:15</td>
      <td>2021-02-24</td>
      <td>3</td>
      <td>[33667174465668, 33667232891012, 3366726808384...</td>
      <td>24</td>
      <td>Proposition</td>
      <td>UKM</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


La gente con hijos compra productos diferentes?


```python
df_exploded2 = df_users_y_orders_sin_na.explode('ordered_items')
df_exploded2['variant_id']=df_exploded2['ordered_items']
```


```python
df_completo_sin_na=df_exploded2.merge(df_inventory, on='variant_id', how='left')
```


```python
df_completo_sin_na.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11636 entries, 0 to 11635
    Data columns (total 22 columns):
     #   Column                 Non-Null Count  Dtype         
    ---  ------                 --------------  -----         
     0   id                     11636 non-null  int64         
     1   user_id                11636 non-null  object        
     2   created_at             11636 non-null  datetime64[us]
     3   order_date             11636 non-null  datetime64[us]
     4   user_order_seq         11636 non-null  int64         
     5   ordered_items          11636 non-null  object        
     6   number_items           11636 non-null  int64         
     7   user_segment           11636 non-null  object        
     8   user_nuts1             11559 non-null  object        
     9   first_ordered_at       11636 non-null  object        
     10  customer_cohort_month  11636 non-null  object        
     11  count_people           11636 non-null  float64       
     12  count_adults           11636 non-null  float64       
     13  count_children         11636 non-null  float64       
     14  count_babies           11636 non-null  float64       
     15  count_pets             11636 non-null  float64       
     16  variant_id             11636 non-null  object        
     17  price                  8937 non-null   float64       
     18  compare_at_price       8937 non-null   float64       
     19  vendor                 8937 non-null   object        
     20  product_type           8937 non-null   object        
     21  tags                   8937 non-null   object        
    dtypes: datetime64[us](2), float64(7), int64(3), object(10)
    memory usage: 2.0+ MB



```python
df_completo_sin_na2 = df_completo_sin_na.dropna(subset=['product_type'])
# Filtrar el DataFrame para obtener solo las filas correspondientes a personas que no tienen hijos
sin_hijos = df_completo_sin_na2[df_completo_sin_na2['count_children'] == 0]

# Calcular el total de compras para este grupo
total_compras_sin_hijos = len(sin_hijos)

# Calcular el porcentaje de cada tipo de producto respecto al total de compras del grupo
porcentaje_por_tipo = (sin_hijos.groupby('product_type').size() / total_compras_sin_hijos) * 100
porcentaje_por_tipo_ordenado = porcentaje_por_tipo.sort_values(ascending=False)
print("Porcentaje de compras por tipo de producto para personas sin hijos:")
print(porcentaje_por_tipo_ordenado)




```

    Porcentaje de compras por tipo de producto para personas sin hijos:
    product_type
    cleaning-products                  11.253143
    tins-packaged-foods                10.100587
    toilet-roll-kitchen-roll-tissue     9.241408
    cereal                              5.217938
    snacks-confectionery                4.274937
    cooking-ingredients                 4.170159
    soft-drinks-mixers                  4.023470
    pasta-rice-noodles                  3.876781
    condiments-dressings                3.730092
    dishwashing                         3.499581
    long-life-milk-substitutes          3.248114
    dental                              2.724225
    cooking-sauces                      2.619447
    hand-soap-sanitisers                2.472758
    spreads                             2.472758
    fabric-softener-freshener           2.200335
    bin-bags                            2.137469
    home-baking                         1.969824
    tea                                 1.927913
    biscuits-crackers                   1.823135
    food-bags-cling-film-foil           1.571668
    haircare                            1.508801
    coffee                              1.299246
    washing-liquid-gel                  1.278290
    bath-shower-gel                     1.278290
    skincare                            1.257334
    washing-capsules                    0.901090
    dog-food                            0.880134
    washing-powder                      0.775356
    cat-food                            0.607712
    period-care                         0.586756
    beer                                0.544845
    deodorant                           0.502934
    delicates-stain-remover             0.481978
    cider                               0.461023
    wine                                0.419111
    pet-care                            0.356245
    drying-ironing                      0.356245
    spirits-liqueurs                    0.272422
    shaving-grooming                    0.251467
    baby-kids-toiletries                0.209556
    water-softener                      0.188600
    nappies-nappy-pants                 0.167645
    sexual-health                       0.125733
    other-hot-drinks                    0.125733
    superfoods-supplements              0.125733
    household-sundries                  0.104778
    medicines-treatments                0.083822
    baby-accessories                    0.083822
    premixed-cocktails                  0.062867
    baby-toddler-food                   0.062867
    suncare                             0.041911
    low-no-alcohol                      0.020956
    adult-incontinence                  0.020956
    dtype: float64



```python
# Filtrar el DataFrame para obtener solo las filas correspondientes a personas que tienen hijos
con_hijos = df_completo_sin_na2[df_completo_sin_na2['count_children'] > 0]

# Calcular el total de compras para este grupo
total_compras_con_hijos = len(con_hijos)

# Calcular el porcentaje de cada tipo de producto respecto al total de compras del grupo
porcentaje_por_tipo_con_hijos = (con_hijos.groupby('product_type').size() / total_compras_con_hijos) * 100

# Ordenar el porcentaje de compras por tipo de producto de mayor a menor
porcentaje_por_tipo_con_hijos_ordenado = porcentaje_por_tipo_con_hijos.sort_values(ascending=False)

print("Porcentaje de compras por tipo de producto para personas con hijos (ordenado de mayor a menor):")
print(porcentaje_por_tipo_con_hijos_ordenado)

```

    Porcentaje de compras por tipo de producto para personas con hijos (ordenado de mayor a menor):
    product_type
    cleaning-products                  14.429772
    tins-packaged-foods                 8.355342
    toilet-roll-kitchen-roll-tissue     7.779112
    dishwashing                         6.170468
    cereal                              4.897959
    snacks-confectionery                4.369748
    cooking-sauces                      4.057623
    hand-soap-sanitisers                3.817527
    cooking-ingredients                 3.289316
    pasta-rice-noodles                  3.025210
    condiments-dressings                3.001200
    dental                              2.496999
    long-life-milk-substitutes          2.472989
    haircare                            2.472989
    soft-drinks-mixers                  2.424970
    spreads                             2.280912
    washing-liquid-gel                  2.136855
    home-baking                         1.992797
    fabric-softener-freshener           1.896759
    bin-bags                            1.800720
    bath-shower-gel                     1.608643
    tea                                 1.416567
    skincare                            1.272509
    biscuits-crackers                   1.224490
    food-bags-cling-film-foil           1.176471
    washing-powder                      1.176471
    baby-kids-toiletries                0.864346
    cat-food                            0.840336
    delicates-stain-remover             0.744298
    pet-care                            0.744298
    wine                                0.672269
    baby-toddler-food                   0.648259
    dog-food                            0.552221
    deodorant                           0.528211
    nappies-nappy-pants                 0.528211
    coffee                              0.504202
    beer                                0.480192
    washing-capsules                    0.408163
    period-care                         0.264106
    shaving-grooming                    0.264106
    spirits-liqueurs                    0.216086
    drying-ironing                      0.144058
    superfoods-supplements              0.096038
    baby-accessories                    0.096038
    cider                               0.048019
    low-no-alcohol                      0.048019
    other-hot-drinks                    0.048019
    baby-milk-formula                   0.048019
    water-softener                      0.048019
    medicine-treatments                 0.048019
    medicines-treatments                0.024010
    household-sundries                  0.024010
    maternity                           0.024010
    dtype: float64



```python
# Reindexar ambos conjuntos de datos para asegurarnos de que tengan las mismas categorías de productos
porcentaje_por_tipo_ordenado = porcentaje_por_tipo_ordenado.reindex(porcentaje_por_tipo_con_hijos_ordenado.index, fill_value=0)

# Restar los porcentajes de cada tipo de producto entre personas sin hijos y personas con hijos
diferencia_porcentaje = porcentaje_por_tipo_ordenado - porcentaje_por_tipo_con_hijos_ordenado

print("Diferencia en porcentaje de compras por tipo de producto (sin hijos - con hijos):")
print(diferencia_porcentaje)


```

    Diferencia en porcentaje de compras por tipo de producto (sin hijos - con hijos):
    product_type
    cleaning-products                 -3.176629
    tins-packaged-foods                1.745245
    toilet-roll-kitchen-roll-tissue    1.462297
    dishwashing                       -2.670887
    cereal                             0.319979
    snacks-confectionery              -0.094811
    cooking-sauces                    -1.438176
    hand-soap-sanitisers              -1.344769
    cooking-ingredients                0.880844
    pasta-rice-noodles                 0.851571
    condiments-dressings               0.728892
    dental                             0.227226
    long-life-milk-substitutes         0.775125
    haircare                          -0.964188
    soft-drinks-mixers                 1.598500
    spreads                            0.191845
    washing-liquid-gel                -0.858565
    home-baking                       -0.022973
    fabric-softener-freshener          0.303577
    bin-bags                           0.336748
    bath-shower-gel                   -0.330353
    tea                                0.511346
    skincare                          -0.015175
    biscuits-crackers                  0.598645
    food-bags-cling-film-foil          0.395197
    washing-powder                    -0.401114
    baby-kids-toiletries              -0.654790
    cat-food                          -0.232624
    delicates-stain-remover           -0.262320
    pet-care                          -0.388053
    wine                              -0.253157
    baby-toddler-food                 -0.585393
    dog-food                           0.327913
    deodorant                         -0.025278
    nappies-nappy-pants               -0.360567
    coffee                             0.795044
    beer                               0.064653
    washing-capsules                   0.492926
    period-care                        0.322650
    shaving-grooming                  -0.012639
    spirits-liqueurs                   0.056336
    drying-ironing                     0.212187
    superfoods-supplements             0.029695
    baby-accessories                  -0.012216
    cider                              0.413003
    low-no-alcohol                    -0.027064
    other-hot-drinks                   0.077714
    baby-milk-formula                 -0.048019
    water-softener                     0.140581
    medicine-treatments               -0.048019
    medicines-treatments               0.059813
    household-sundries                 0.080768
    maternity                         -0.024010
    dtype: float64


Las diferencias no son significativas

Existen tipos de productos que se compran mas en diferentes estaciones del año?


```python
df_exploded_orders = df_orders.explode('ordered_items')
df_exploded_orders=df_exploded_orders.rename(columns={'ordered_items': 'variant_id'})
df_inventory_orders=df_exploded_orders.merge( df_inventory, on='variant_id', how='right')
```


```python
df_inventory_orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 92617 entries, 0 to 92616
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   id                92361 non-null  float64       
     1   user_id           92361 non-null  object        
     2   created_at        92361 non-null  datetime64[us]
     3   order_date        92361 non-null  datetime64[us]
     4   user_order_seq    92361 non-null  float64       
     5   variant_id        92617 non-null  object        
     6   number_items      92361 non-null  float64       
     7   price             92617 non-null  float64       
     8   compare_at_price  92617 non-null  float64       
     9   vendor            92617 non-null  object        
     10  product_type      92617 non-null  object        
     11  tags              92617 non-null  object        
    dtypes: datetime64[us](2), float64(5), object(5)
    memory usage: 8.5+ MB


Evolucion de las ventas por cada tipo de producto


```python
import matplotlib.pyplot as plt

# 1. Agrupar por trimestre y categoría de producto
df_inventory_orders['trimestre'] = df_inventory_orders['order_date'].dt.to_period('Q')
grouped = df_inventory_orders.groupby(['trimestre', 'product_type'])

# 2. Calcular el recuento de compras para cada categoría de producto en cada trimestre
compras_por_trimestre = grouped.size().unstack(fill_value=0)

# 3. Calcular las ventas totales por trimestre
ventas_totales_por_trimestre = compras_por_trimestre.sum(axis=1)

# 4. Calcular el porcentaje de ventas para cada categoría de producto respecto a las ventas totales por trimestre
porcentaje_ventas_por_categoria = compras_por_trimestre.div(ventas_totales_por_trimestre, axis=0)

# 5. Dividir las categorías en grupos de 5
categorias = porcentaje_ventas_por_categoria.columns
grupos_categorias = [categorias[i:i+5] for i in range(0, len(categorias), 5)]

# 6. Convertir períodos a marcas de tiempo
porcentaje_ventas_por_categoria.index = porcentaje_ventas_por_categoria.index.to_timestamp()

# 7. Crear un gráfico de serie temporal para cada grupo de categorías
for grupo in grupos_categorias:
    plt.figure(figsize=(12, 6))
    for categoria in grupo:
        plt.plot(porcentaje_ventas_por_categoria.index, porcentaje_ventas_por_categoria[categoria], label=f'Porcentaje de Ventas de {categoria} respecto a Ventas Totales')
    
    plt.title('Porcentaje de Ventas de Categorías respecto a Ventas Totales por Trimestre')
    plt.xlabel('Trimestre')
    plt.ylabel('Porcentaje de Ventas')
    plt.legend()
    plt.grid(True)
    plt.show()
  

```
