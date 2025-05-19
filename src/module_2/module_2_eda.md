# MODULO 2: EXPLORATORY DATA ANALYSIS
## 1. Recolecta de Datos y análisis por separado de tablas


```python
import pandas as pd
import os
```


```python
data_dir = "../../data"

files = {
    "orders": "orders.parquet",
    "regulars": "regulars.parquet",
    "abandoned_carts": "abandoned_carts.parquet",
    "inventory": "inventory.parquet",
    "users": "users.parquet"
}

dfs = {}
for name, file in files.items():
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        dfs[name] = pd.read_parquet(path)
    else:
        print(f"Archivo no encontrado: {path}")

for name, file in files.items():
    path = os.path.join(data_dir, file)
    dfs[name] = pd.read_parquet(path)
```


```python
for name, df in dfs.items():
    print(name)
    print(df.shape)
```

    orders
    (8773, 6)
    regulars
    (18105, 3)
    abandoned_carts
    (5457, 4)
    inventory
    (1733, 6)
    users
    (4983, 10)


Voy a hacer primero un análisis para ver que contiene cada uno de los distintos datasets
### Primeros análisis
1. Orders


```python
df = dfs["orders"]

def validar_user_order_seq(grupo):
    n = len(grupo)
    secuencia = set(grupo["user_order_seq"])
    return (max(secuencia) == n) and (secuencia == set(range(1, n+1)))

validez = df.groupby("user_id").apply(validar_user_order_seq)

print("Usuarios con secuencia válida:", validez.sum())
print("Usuarios con secuencia inválida:", (~validez).sum())
```


```python
df.info()
```


```python
num_orders = df.groupby(["user_id", "created_at"]).ngroups
print("Number of unique orders:", num_orders)
id_count = df["id"].nunique()
print("Number of unique id's:", id_count)
```


```python
coinciden = df["created_at"].dt.date == df["order_date"].dt.date
print("Coinciden", coinciden.sum())
print("No coinciden", (~coinciden).sum())
df["created_at_hora"] = df["created_at"].dt.time
display(df[~coinciden].sort_values(by="created_at_hora", ascending=True).head(5))
```


```python
errores = df[~coinciden].copy()
errores["diferencia_dias"] = (errores["created_at"] - errores["order_date"]).dt.days
print(errores["diferencia_dias"].value_counts())
errores_max = max(errores["created_at"].dt.time)
no_errores_min = min(df[coinciden]["created_at"].dt.time)
print(errores_max)
print(no_errores_min)
```

Aquí observamos que, segun habímaos supuesto, en esta tabla aparece la siguiente información:
* el identificador del usuario tantas veces como pedidos haya hecho
* el pedido al qual hace referencia cada una de las filas (si un usuario ha hecho 3 pedidos aparecerán en cada fila con un 1, 2 o 3)
* las referencias de los productos comprados en cada uno de los pedidos.
* la variable order_date es el dia en que queda registrada como pedida la comanda y esta coincide con la fecha en la que se creo la comanda, en 51 casos, la comanda fue creada en la madrugada (concretamente entre las 00:25 y las 00:58) y queda registrada como que el pedido se inicio el dia antes. Por lo tanto, nos vamos a quedar solo con la variable created_at en el futuro.
* no hay valores nulos
  

2. Regulars


```python
df = dfs["regulars"]
df.info()
```


```python
df_grouped = df.groupby(["user_id", "created_at"])["variant_id"].agg(list).reset_index()
display(df_grouped)
```

Aquí observamos que, esta tabla hace referncia para cada usuario en el momento que ha creado una comanda, que identificador de producto ha añadido a la lista. Se comprueba, segun lo esperado que:
* no hay valores nulos
* si agrupamos por comanda (usuario + hora de creacion de la comanda), hay un total de 1878 comandas diferentes.
* vemos que no hay el mismo numero de comandas que en el dataset orders
Puede que esta tabla haga referencia a los productos que mas compran los usuarios que mas frecuentan la pagina. Se debería preguntar que es exactamente esta tabla.

3. abandoned_carts


```python
df = dfs["abandoned_carts"]
df.info()
display(df)
```


```python
id_comun = dfs["orders"]["id"].isin(dfs["abandoned_carts"]["id"])
print("Numero de identificadores de operaciones comunes entre abandoned_carts y orders: ", id_comun.sum())
```

Interesante obersvacion, vemos que todos los id's son nuevos en esta tabla respecto la tabla de orders. Podemos suponer por el nombre del dataset que por un lado tenemos las orders que se han hecho y pagado y por otro las comandas que se han hecho, se han añadido los paquetes pero a la hora de la verdad no se ha realizado el pago y se ha abandonado el carro de la compra.

4. Inventory


```python
df = dfs["inventory"]
df.info()
df.head()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 df = dfs["inventory"]
          2 df.info()
          3 df.head()


    KeyError: 'inventory'



```python
df["tags_str"] = df["tags"].apply(lambda x: ", ".join(x))
df.groupby(["product_type", "tags_str"])["compare_at_price"].nunique().sort_values()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 df["tags_str"] = df["tags"].apply(lambda x: ", ".join(x))
          2 df.groupby(["product_type", "tags_str"])["compare_at_price"].nunique().sort_values()


    NameError: name 'df' is not defined



```python
coincide = df["price"] == df["compare_at_price"]
print("Numero de precios que coinciden con su comparativa: ", coincide.sum())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 1
    ----> 1 coincide = df["price"] == df["compare_at_price"]
          2 print("Numero de precios que coinciden con su comparativa: ", coincide.sum())


    NameError: name 'df' is not defined


En el caso del dataset inventory. Hace referencia a los distintos productos que tienen y las caracteristicas de cada uno de estos. La unica varaible de la columna que no acabo de interpretar es compare_at_price. 
Inicialmente he supuesto que podria ser el precio previo al descuento aplicado, sin embargo solo hay 103 variant_id que tienen el mismo valor en ambas variables.
Tampoco ha reusltado ser una variable haciendo referencia a alguna media hecha con el resto de predductos del mismo tipo, ni del mismo tipo y tag. He realizado la prueba para ambos casos.

5. users


```python
df = dfs["users"]
df.info()
display(df)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[6], line 1
    ----> 1 df = dfs["users"]
          2 df.info()
          3 display(df)


    KeyError: 'users'



```python
df["user_id"].nunique()
no_nulos = df["count_people"].notna()
display(df[no_nulos])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 df["user_id"].nunique()
          2 no_nulos = df["count_people"].notna()
          3 display(df[no_nulos])


    NameError: name 'df' is not defined


No se ve una relación entre las variables de contador no nulas y las variables que si poseemos. Ademas la mayoria de estas varibales tienen valor nulo. Lo único que observamos es que cuando el valor es no nulo entonces siempre es no nulo. Es posible que cuando juntemos las diferentes tablas enconctremos mas información de porque tenemos de unas la información y de otras no.
Como observación,cabe destacara que la variable customer_cohort_month es el primer dia de mes de la variable first_ordered_at.
Se debería pedir mas información sobre las variables de count para ver a que hacen referencia.

## Union de las distintas tablas segun convenga
#### COMO UNIREMOS:
1. tabla1,  orders y abandoned_carts: una debajo de la otra, con un append, añadiremos una variable binaria donde los 1 seran los abandoned cart y los 0 seran los orders. Suponemos que la target variable será esta variable binario que nos diga a nosotros si el usuario ha realizado la compra finalmente o no. Ademas añadimos que la variable user_order_seq para los abandoned_cart sea 0. (esto ademas ya definiria el binario que hemos comentado posteriormente
2. tabla2, unimos tabla1 con users: hacemos un join de la tabla1 creada con el dataset users uniendo por user_id. De tal manera que tendremos añadida a nuestra tabla toda la info por user
3. tabla3, tabla2 y regulars: hacemos un indicador de las relaciones user/producto que estan en regulars (se suelen consumir) y añadimos un indicador en la tabla3 resultante que nos diga si un usuario en concreto con una compra en concreto contiene un producto que suele comprar regularmente, le llamaremos variable "regular"
4. tabla4, tabla3 y inventory: considero que la manera mas visual e identificativa que añadir seria una suma de todos los precios y de los compared_price de cada una de las entradas de las orders y abandoned_carts que tenemos. De esta manera tendremos el precio de la compra entera y el compare_price de la compra entera. De momento no añadiré todos vendors y los product_types, necesitaria mas contexto de la situacion y de lo que queremos saber con estas variables.


```python
dfs["orders"]["order_placed"] = 1
dfs["abandoned_carts"]["order_placed"] = 0
df_abandoned = dfs["abandoned_carts"]
dfs["orders"] = dfs["orders"].rename(columns={'ordered_items': 'variant_id'})
df_orders=dfs["orders"].drop(['created_at_hora', 'order_date'], axis=1)
```


```python
df_tabla1=pd.concat([df_orders, df_abandoned], ignore_index=True)
df_tabla1["user_order_seq"]=df_tabla1["user_order_seq"].fillna(0)
display(df_tabla1)
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
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>order_placed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>1.0</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>1.0</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>1.0</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>1.0</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>1.0</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14225</th>
      <td>22233840976004</td>
      <td>2e989bfdec87ef55ea464a529f323ff53dad2a2fc48655...</td>
      <td>2022-03-13 14:11:15</td>
      <td>0.0</td>
      <td>[34284950192260, 39466620911748]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14226</th>
      <td>22233843171460</td>
      <td>b2d867b982b14ca517f27c4ced727c8a25c01b96ebbd96...</td>
      <td>2022-03-13 14:11:36</td>
      <td>0.0</td>
      <td>[39536607395972, 39506484461700]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14227</th>
      <td>22233843531908</td>
      <td>220aafc0749f209b3f0f7cfe4134a5136815d48f0bbd9a...</td>
      <td>2022-03-13 14:11:41</td>
      <td>0.0</td>
      <td>[39482337624196, 39544243650692]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14228</th>
      <td>22233846218884</td>
      <td>a4da55d51052411e54f98e1b90b19843121866abeaea76...</td>
      <td>2022-03-13 14:12:09</td>
      <td>0.0</td>
      <td>[34415989325956, 33667297017988, 3948233762419...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14229</th>
      <td>22233846317188</td>
      <td>c0e740ecabe7bd19eaed35b5ea9be7bc80c15f32124712...</td>
      <td>2022-03-13 14:12:10</td>
      <td>0.0</td>
      <td>[34284950519940, 39459281174660, 39482337558660]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>14230 rows × 6 columns</p>
</div>



```python
df_tabla2=pd.merge(df_tabla1, dfs["users"], on='user_id', how='left')
display(df_tabla2)
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
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>order_placed</th>
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
      <td>1.0</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
      <td>1</td>
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
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>1.0</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
      <td>1</td>
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
      <th>2</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>1.0</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
      <td>1</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>1.0</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
      <td>1</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01 00:00:00</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>1.0</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
      <td>1</td>
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
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14225</th>
      <td>22233840976004</td>
      <td>2e989bfdec87ef55ea464a529f323ff53dad2a2fc48655...</td>
      <td>2022-03-13 14:11:15</td>
      <td>0.0</td>
      <td>[34284950192260, 39466620911748]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKE</td>
      <td>2022-01-19 11:22:29</td>
      <td>2022-01-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14226</th>
      <td>22233843171460</td>
      <td>b2d867b982b14ca517f27c4ced727c8a25c01b96ebbd96...</td>
      <td>2022-03-13 14:11:36</td>
      <td>0.0</td>
      <td>[39536607395972, 39506484461700]</td>
      <td>0</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2022-02-01 19:07:39</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14227</th>
      <td>22233843531908</td>
      <td>220aafc0749f209b3f0f7cfe4134a5136815d48f0bbd9a...</td>
      <td>2022-03-13 14:11:41</td>
      <td>0.0</td>
      <td>[39482337624196, 39544243650692]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKC</td>
      <td>2022-02-26 12:27:31</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14228</th>
      <td>22233846218884</td>
      <td>a4da55d51052411e54f98e1b90b19843121866abeaea76...</td>
      <td>2022-03-13 14:12:09</td>
      <td>0.0</td>
      <td>[34415989325956, 33667297017988, 3948233762419...</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2022-03-07 13:12:33</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14229</th>
      <td>22233846317188</td>
      <td>c0e740ecabe7bd19eaed35b5ea9be7bc80c15f32124712...</td>
      <td>2022-03-13 14:12:10</td>
      <td>0.0</td>
      <td>[34284950519940, 39459281174660, 39482337558660]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2022-03-07 15:59:54</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14230 rows × 15 columns</p>
</div>



```python
df_tabla2["created_at_day"]=df_tabla2["created_at"].dt.date
```


```python
dfs["regulars"]["variant_id"]=dfs["regulars"]["variant_id"].astype(str)
df_regulars = dfs["regulars"]
df_regulars = df_regulars.groupby(["user_id", "created_at"])["variant_id"].agg(list).reset_index()
df_regulars["created_at_day"]=df_regulars["created_at"].dt.date
df_regulars=df_regulars.rename(columns={'variant_id': 'lista_variant_id'})
```


```python
display(df_regulars)
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
      <th>created_at</th>
      <th>lista_variant_id</th>
      <th>created_at_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>004b3e3cb9a9f5b0974ce4179db394057c72e7a82077bf...</td>
      <td>2021-12-21 21:48:05</td>
      <td>[33667274997892]</td>
      <td>2021-12-21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>005743eefffa4ce840608c4f47b8c548b134d89be5c390...</td>
      <td>2021-06-09 08:40:19</td>
      <td>[34081589887108, 34519123951748]</td>
      <td>2021-06-09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0074992079c1836c6509eec748a973dc97388b4877e770...</td>
      <td>2020-05-18 20:35:53</td>
      <td>[33667222896772, 33826414526596, 3382641364186...</td>
      <td>2020-05-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00ecced73edb11d4bab08e794656dcf9d9b89ea89c5918...</td>
      <td>2021-01-23 09:01:42</td>
      <td>[33667283648644, 33667214246020, 34221708083332]</td>
      <td>2021-01-23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>014301579c18e7c7f034e544ab3d4ee235ef2de43ee5db...</td>
      <td>2021-06-15 09:09:37</td>
      <td>[33803538432132, 34221708771460, 3366718265766...</td>
      <td>2021-06-15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1873</th>
      <td>fffc1d81bbde7ce58c679994aa863323198ff6a3afef4c...</td>
      <td>2021-06-05 16:45:25</td>
      <td>[34488547475588]</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>fffd9f989509e36d1fc3e3e53627d6341482f385052a03...</td>
      <td>2021-10-28 13:23:35</td>
      <td>[33826413019268, 34465293402244, 3382646040179...</td>
      <td>2021-10-28</td>
    </tr>
    <tr>
      <th>1875</th>
      <td>fffd9f989509e36d1fc3e3e53627d6341482f385052a03...</td>
      <td>2022-01-20 09:27:43</td>
      <td>[39590266536068, 33826460401796, 3431785011622...</td>
      <td>2022-01-20</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>fffd9f989509e36d1fc3e3e53627d6341482f385052a03...</td>
      <td>2022-01-20 09:39:11</td>
      <td>[33826460401796, 39590266536068, 3382641301926...</td>
      <td>2022-01-20</td>
    </tr>
    <tr>
      <th>1877</th>
      <td>fffd9f989509e36d1fc3e3e53627d6341482f385052a03...</td>
      <td>2022-02-04 10:56:52</td>
      <td>[33826413019268, 34465293107332, 3932091200320...</td>
      <td>2022-02-04</td>
    </tr>
  </tbody>
</table>
<p>1878 rows × 4 columns</p>
</div>



```python
df_tabla3=pd.merge(df_tabla2, df_regulars, on=['user_id', 'created_at_day'], how='left')
display(df_tabla3)
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
      <th>created_at_x</th>
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>order_placed</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>created_at_day</th>
      <th>created_at_y</th>
      <th>lista_variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>1.0</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
      <td>1</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2020-04-30</td>
      <td>2020-04-30 13:09:27</td>
      <td>[33618909495428, 33618981421188, 3361886017958...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>1.0</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
      <td>1</td>
      <td>Proposition</td>
      <td>UKM</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2020-04-30</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>1.0</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
      <td>1</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2020-04-30</td>
      <td>2020-04-30 17:06:48</td>
      <td>[33618998853764, 33618846580868, 3361899659277...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>1.0</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
      <td>1</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01 00:00:00</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2020-05-01</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>1.0</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
      <td>1</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-01 00:00:00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2020-05-03</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14348</th>
      <td>22233840976004</td>
      <td>2e989bfdec87ef55ea464a529f323ff53dad2a2fc48655...</td>
      <td>2022-03-13 14:11:15</td>
      <td>0.0</td>
      <td>[34284950192260, 39466620911748]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKE</td>
      <td>2022-01-19 11:22:29</td>
      <td>2022-01-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-03-13</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14349</th>
      <td>22233843171460</td>
      <td>b2d867b982b14ca517f27c4ced727c8a25c01b96ebbd96...</td>
      <td>2022-03-13 14:11:36</td>
      <td>0.0</td>
      <td>[39536607395972, 39506484461700]</td>
      <td>0</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2022-02-01 19:07:39</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-03-13</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14350</th>
      <td>22233843531908</td>
      <td>220aafc0749f209b3f0f7cfe4134a5136815d48f0bbd9a...</td>
      <td>2022-03-13 14:11:41</td>
      <td>0.0</td>
      <td>[39482337624196, 39544243650692]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKC</td>
      <td>2022-02-26 12:27:31</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-03-13</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14351</th>
      <td>22233846218884</td>
      <td>a4da55d51052411e54f98e1b90b19843121866abeaea76...</td>
      <td>2022-03-13 14:12:09</td>
      <td>0.0</td>
      <td>[34415989325956, 33667297017988, 3948233762419...</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2022-03-07 13:12:33</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-03-13</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14352</th>
      <td>22233846317188</td>
      <td>c0e740ecabe7bd19eaed35b5ea9be7bc80c15f32124712...</td>
      <td>2022-03-13 14:12:10</td>
      <td>0.0</td>
      <td>[34284950519940, 39459281174660, 39482337558660]</td>
      <td>0</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2022-03-07 15:59:54</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-03-13</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14353 rows × 18 columns</p>
</div>



```python
df_tabla3.drop(c
```

cuando siga:
pasar a string la variable variant, reconstruir el created_at para que solo se haga el join por el dia no solo la hora


```python

```
