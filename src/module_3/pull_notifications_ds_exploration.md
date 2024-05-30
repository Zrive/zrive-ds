# Push notifications DS

## Milestone 1: Exploration phase

### 1. Filter data:

Get dataset feature_frame_20210304.csv and filter by orders with at least 5 items:


```python
import boto3
import pandas as pd
import numpy as np
import fastparquet
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from sklearn import linear_model
from typing import Tuple
```


```python
local_file_path = '/home/ebacigalupe/zrive-ds/zrive-ds/src/module_3/feature_frame.csv'  

feature_frame = pd.read_csv(local_file_path)
feature_frame.head()
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
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
feature_frame.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
print(f"Unique number of orders:", feature_frame['order_id'].nunique())
```

    Unique number of orders: 3446


Define columns of the dataset by type: 
- Information columns
- Label
- Features
- Categorical
- Binary
- Numerical

I add this to my code since I saw it in Guille's solution and I liked it.


```python
info_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
label_col = 'outcome'
features_cols = [col for col in feature_frame.columns if col not in info_cols + [label_col]]

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols = [col for col in features_cols if col not in categorical_cols + binary_cols]
```

Steps to filter:
1. Group feature_frame by order_id and sum outcome. Only purchased products have outcome = 1.
2. Filter the resulting dataset by number of items per order >= 5.
3. Join feature_frame with feature_frame_orders_bt5 to keep only orders with >= 5 items.


```python
feature_frame_orders = feature_frame.groupby('order_id').outcome.sum()
feature_frame_orders_bt5 = feature_frame_orders[feature_frame_orders >= 5].index
feature_frame_filtered = feature_frame.loc[lambda x: x.order_id.isin(feature_frame_orders_bt5)]
print(f"Unique number of orders:", feature_frame_filtered['order_id'].nunique())

```

    Unique number of orders: 2603



```python
feature_frame_filtered.head()
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
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



Let's briefly analize how orders behave over time:


```python
feature_frame_filtered['order_date'] = pd.to_datetime(feature_frame_filtered['order_date'])
daily_orders = feature_frame_filtered.groupby('order_date').order_id.nunique()
```

    /tmp/ipykernel_1062/3174384821.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      feature_frame_filtered['order_date'] = pd.to_datetime(feature_frame_filtered['order_date'])



```python
plt.figure(figsize=(10, 6))
daily_orders.plot(marker='o', linestyle='-')
plt.title('Number of Orders per Day')
plt.xlabel('Order Date')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_15_0.png)
    


It is a must to make sure to do a temporal split to avoid data leakage. No order should be split between train and test datasets. To do so, let's compute a cummulative sum of the daily orders:


```python
daily_orders_acc = daily_orders.cumsum()/daily_orders.sum()
```


```python
train_validation = daily_orders_acc[daily_orders_acc <= 0.7].idxmax()
validation_test = daily_orders_acc[daily_orders_acc <= 0.9].idxmax()
```


```python
train_feature_frame = feature_frame_filtered[feature_frame_filtered.order_date <= train_validation]
val_feature_frame = feature_frame_filtered[(feature_frame_filtered.order_date > train_validation) &(feature_frame_filtered.order_date <= validation_test)]
test_feature_frame = feature_frame_filtered[feature_frame_filtered.order_date > validation_test]
```

### 2. PoC - Linear model:

The goal is to build a machine learning model that, given a user and a product, predicts if the
user would purchase it if they were buying with us at that point in time.

A linear model must be applied to feature_frame_filtered using sklearn linear_model.

Model training:


```python
X_train = train_feature_frame.drop(label_col, axis=1)
y_train = train_feature_frame[label_col]

X_val = val_feature_frame.drop(label_col, axis=1)
y_val = val_feature_frame[label_col]

X_test = test_feature_frame.drop(label_col, axis=1)
y_test = test_feature_frame[label_col]

```

The training will start only with non categorical columns:


```python
train_cols = numerical_cols + binary_cols
```

#### Baseline

A simple model must be set up to act as a threshold againg more complex models. For this dataset, global popularity feature will be used as baseline. 


```python
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in /home/ebacigalupe/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages (1.4.2)
    Requirement already satisfied: scipy>=1.6.0 in /home/ebacigalupe/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages (from scikit-learn) (1.13.0)
    Requirement already satisfied: numpy>=1.19.5 in /home/ebacigalupe/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /home/ebacigalupe/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages (from scikit-learn) (3.4.0)
    Requirement already satisfied: joblib>=1.2.0 in /home/ebacigalupe/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages (from scikit-learn) (1.4.0)



```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use('fast')
```


```python
def plot_metrics(model_name: str, y_pred: pd.Series, y_test: pd.Series, target_precision: float=0.05, figure: Tuple[matplotlib.figure.Figure, np.array]= None):
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(14,7))
    else:
        fig, ax = figure
    
    ax[0].plot(recall_, precision_, label=f'{model_name}; (AUC = {pr_auc:.2f})')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title('Precision-Recall Curve')
    ax[0].legend(loc='lower right')

    ax[1].plot(fpr, tpr, label=f'(AUC = {roc_auc:.2f})')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate ')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc='upper center')
```


```python
plot_metrics("Popularity baseline", y_pred=val_feature_frame["global_popularity"], y_test=val_feature_frame[label_col])
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_32_0.png)
    


#### Ridge regularization


```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
```


```python
lr_push_train_aucs = []
lr_push_val_aucs = []
lr_push_train_ce = []
lr_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]
for c in cs:
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2' if c else "none", C=c if c else 1.0)
    )
    lr.fit(X_train[train_cols], y_train)
    train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"LR; C = {c}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"LR; C = {c}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))

```


    ---------------------------------------------------------------------------

    InvalidParameterError                     Traceback (most recent call last)

    Cell In[18], line 18
         13 for c in cs:
         14     lr = make_pipeline(
         15         StandardScaler(),
         16         LogisticRegression(penalty='l2' if c else "none", C=c if c else 1.0)
         17     )
    ---> 18     lr.fit(X_train[train_cols], y_train)
         19     train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
         20     plot_metrics(f"LR; C = {c}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:1474, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1467     estimator._validate_params()
       1469 with config_context(
       1470     skip_parameter_validation=(
       1471         prefer_skip_nested_validation or global_skip_validation
       1472     )
       1473 ):
    -> 1474     return fit_method(estimator, *args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/pipeline.py:475, in Pipeline.fit(self, X, y, **params)
        473     if self._final_estimator != "passthrough":
        474         last_step_params = routed_params[self.steps[-1][0]]
    --> 475         self._final_estimator.fit(Xt, y, **last_step_params["fit"])
        477 return self


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:1467, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1462 partial_fit_and_fitted = (
       1463     fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
       1464 )
       1466 if not global_skip_validation and not partial_fit_and_fitted:
    -> 1467     estimator._validate_params()
       1469 with config_context(
       1470     skip_parameter_validation=(
       1471         prefer_skip_nested_validation or global_skip_validation
       1472     )
       1473 ):
       1474     return fit_method(estimator, *args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:666, in BaseEstimator._validate_params(self)
        658 def _validate_params(self):
        659     """Validate types and values of constructor parameters
        660 
        661     The expected type and values must be defined in the `_parameter_constraints`
       (...)
        664     accepted constraints.
        665     """
    --> 666     validate_parameter_constraints(
        667         self._parameter_constraints,
        668         self.get_params(deep=False),
        669         caller_name=self.__class__.__name__,
        670     )


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/utils/_param_validation.py:95, in validate_parameter_constraints(parameter_constraints, params, caller_name)
         89 else:
         90     constraints_str = (
         91         f"{', '.join([str(c) for c in constraints[:-1]])} or"
         92         f" {constraints[-1]}"
         93     )
    ---> 95 raise InvalidParameterError(
         96     f"The {param_name!r} parameter of {caller_name} must be"
         97     f" {constraints_str}. Got {param_val!r} instead."
         98 )


    InvalidParameterError: The 'penalty' parameter of LogisticRegression must be a str among {'elasticnet', 'l2', 'l1'} or None. Got 'none' instead.



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_35_1.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_35_2.png)
    


#### Lasso regularization 


```python
lr_push_train_aucs = []
lr_push_val_aucs = []
lr_push_train_ce = []
lr_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]
cmap = plt.get_cmap('Paired')
for c in cs:
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1" if c else "none", C=c if c else 1.0, solver ="saga")
    )
    lr.fit(X_train[train_cols], y_train)
    train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"LR; C = {c}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"LR; C = {c}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))

plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))

```


    ---------------------------------------------------------------------------

    InvalidParameterError                     Traceback (most recent call last)

    Cell In[19], line 19
         14 for c in cs:
         15     lr = make_pipeline(
         16         StandardScaler(),
         17         LogisticRegression(penalty="l1" if c else "none", C=c if c else 1.0, solver ="saga")
         18     )
    ---> 19     lr.fit(X_train[train_cols], y_train)
         20     train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
         21     plot_metrics(f"LR; C = {c}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:1474, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1467     estimator._validate_params()
       1469 with config_context(
       1470     skip_parameter_validation=(
       1471         prefer_skip_nested_validation or global_skip_validation
       1472     )
       1473 ):
    -> 1474     return fit_method(estimator, *args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/pipeline.py:475, in Pipeline.fit(self, X, y, **params)
        473     if self._final_estimator != "passthrough":
        474         last_step_params = routed_params[self.steps[-1][0]]
    --> 475         self._final_estimator.fit(Xt, y, **last_step_params["fit"])
        477 return self


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:1467, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1462 partial_fit_and_fitted = (
       1463     fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
       1464 )
       1466 if not global_skip_validation and not partial_fit_and_fitted:
    -> 1467     estimator._validate_params()
       1469 with config_context(
       1470     skip_parameter_validation=(
       1471         prefer_skip_nested_validation or global_skip_validation
       1472     )
       1473 ):
       1474     return fit_method(estimator, *args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/base.py:666, in BaseEstimator._validate_params(self)
        658 def _validate_params(self):
        659     """Validate types and values of constructor parameters
        660 
        661     The expected type and values must be defined in the `_parameter_constraints`
       (...)
        664     accepted constraints.
        665     """
    --> 666     validate_parameter_constraints(
        667         self._parameter_constraints,
        668         self.get_params(deep=False),
        669         caller_name=self.__class__.__name__,
        670     )


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-djxvuVXx-py3.11/lib/python3.11/site-packages/sklearn/utils/_param_validation.py:95, in validate_parameter_constraints(parameter_constraints, params, caller_name)
         89 else:
         90     constraints_str = (
         91         f"{', '.join([str(c) for c in constraints[:-1]])} or"
         92         f" {constraints[-1]}"
         93     )
    ---> 95 raise InvalidParameterError(
         96     f"The {param_name!r} parameter of {caller_name} must be"
         97     f" {constraints_str}. Got {param_val!r} instead."
         98 )


    InvalidParameterError: The 'penalty' parameter of LogisticRegression must be a str among {'elasticnet', 'l2', 'l1'} or None. Got 'none' instead.



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_37_1.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_37_2.png)
    


#### Coefficients weights:


```python
lr = Pipeline([("standard_scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1e-6))])
lr.fit(X_train[train_cols], y_train)
lr_coeff_l2 = pd.DataFrame({"features": train_cols, "importance": np.abs(lr.named_steps["lr"].coef_[0]),
                            "regularisation": ["l2"] * len(train_cols)})
lr_coeff_l2 = lr_coeff_l2.sort_values('importance', ascending=True)

lr = Pipeline([("standard_scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l1", C=1e-4, solver="saga"))])
lr.fit(X_train[train_cols], y_train)
lr_coeff_l1 = pd.DataFrame({"features": train_cols, "importance": np.abs(lr.named_steps["lr"].coef_[0]),
                            "regularisation": "l1"})
lr_coeff_l1 = lr_coeff_l1.sort_values('importance', ascending=True)
```


```python
lr_coeffs = pd.concat([lr_coeff_l2, lr_coeff_l1])
lr_coeffs["features"] = pd.Categorical(lr_coeffs["features"])
lr_coeffs = lr_coeffs.sort_values(by=["importance"])
order_columns = lr_coeff_l2.sort_values(by="importance", ascending=False)["features"]
sns.barplot(data=lr_coeffs, x="importance", y="features", hue="regularisation", order=order_columns)
```




    <Axes: xlabel='importance', ylabel='features'>




    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_40_1.png)
    


The model is going to be trained again without non important variables based on L1


```python
reduced_cols = ['ordered_before', 'abandoned_before', 'global_popularity']

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

lrs = [
    make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1e-6)
    ),
    make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", C=1e-4, solver="saga")
    ),
]

names = ['Ridge C=1e-6', "Lasso C=1e-4"]
for name, lr in zip(names, lrs):
    lr.fit(X_train[reduced_cols], y_train)
    train_proba = lr.predict_proba(X_train[reduced_cols])[:, 1]
    plot_metrics(name, y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = lr.predict_proba(X_val[reduced_cols])[:, 1]
    plot_metrics(name, y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_42_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_42_1.png)
    


A linear model can be trained over few features and get same results as a model with several features.

#### Categorical encoding:


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

categorical_preprocessors = [
    ("drop", "drop"),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    (
        "one_hot",
        OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False),
    ),
    ("target", TargetEncoder(target_type="continuous"))
]

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

extended_cols = reduced_cols + categorical_cols

for name, categorical_preprocessors in categorical_preprocessors:
    preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", reduced_cols),
            ("categorical", categorical_preprocessors, categorical_cols),
        ]
    )
    lr = make_pipeline(
        preprocessor,
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1e-6)
    )

    lr.fit(X_train[extended_cols], y_train)
    train_proba = lr.predict_proba(X_train[extended_cols])[:, 1]
    plot_metrics(name, y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = lr.predict_proba(X_val[extended_cols])[:, 1]
    plot_metrics(name, y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_45_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_45_1.png)
    


### 3. PoC - Non-Linear model:

Need to also take into account non-linear models to try to improve previous outcomes.

#### Random forest:


```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
```


```python
rf_push_train_aucs = []
rf_push_val_aucs = []
rf_push_train_ce = []
rf_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

n_trees_grid = [5, 25, 50, 100]
for n_trees in n_trees_grid:
    rf = RandomForestClassifier(n_trees)
    rf.fit(X_train[train_cols], y_train)
    train_proba = rf.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"RF; n_trees = {n_trees}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"RF; n_trees = {n_trees}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_50_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_50_1.png)
    



```python
rf = RandomForestClassifier(100)
rf.fit(X_train[train_cols], y_train)

feature_importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({"features": train_cols, "importance": feature_importances})

feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance_df, x="importance", y="features")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.show()
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_51_0.png)
    


Let's try retraining the model with the most importante features


```python
train_important_cols = feature_importance_df["features"].iloc[:15]
```


```python
rf_push_train_aucs = []
rf_push_val_aucs = []
rf_push_train_ce = []
rf_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

n_trees_grid = [5, 25, 50, 100]
for n_trees in n_trees_grid:
    rf = RandomForestClassifier(n_trees)
    rf.fit(X_train[train_important_cols], y_train)
    train_proba = rf.predict_proba(X_train[train_important_cols])[:, 1]
    plot_metrics(f"RF; n_trees = {n_trees}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

    val_proba = rf.predict_proba(X_val[train_important_cols])[:, 1]
    plot_metrics(f"RF; n_trees = {n_trees}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_54_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_54_1.png)
    


#### Gradient boosting trees:


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
gbt_push_train_aucs = []
gbt_push_val_aucs = []
gbt_push_train_ce = []
gbt_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

n_trees_grid = [5, 25, 50, 100]
for lr in [0.05, 0.1]:
    for depth in [1, 3, 5]:
        for n_trees in n_trees_grid:
            gbt = GradientBoostingClassifier(
                learning_rate=lr, max_depth=depth, n_estimators=n_trees
            )
            gbt.fit(X_train[train_cols], y_train)
            train_proba = gbt.predict_proba(X_train[train_cols])[:, 1]
            plot_metrics(f"gbt; LR = {lr}; MD = {depth}; n_trees = {n_trees}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

            val_proba = gbt.predict_proba(X_val[train_cols])[:, 1]
            plot_metrics(f"gbt; LR = {lr}; MD = {depth}; n_trees = {n_trees}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_57_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_57_1.png)
    



```python
lr_best = 0.1
max_depth_best = 3
n_estimators_best = 100
```


```python
gbt = GradientBoostingClassifier(
        learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
)
gbt.fit(X_train[train_cols], y_train)

feature_importances_gbt = gbt.feature_importances_

feature_importance_df_gbt = pd.DataFrame({"features": train_cols, "importance": feature_importances_gbt})

feature_importance_df_gbt = feature_importance_df_gbt.sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance_df_gbt, x="importance", y="features")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.show()
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_59_0.png)
    



```python
train_important_cols_gbt = (
    feature_importance_df_gbt.loc[feature_importance_df_gbt["importance"] > 0]
    .sort_values(by="importance", ascending=False)["features"]
    .tolist()
)

train_important_cols_gbt = train_important_cols_gbt[:15]
```


```python
gbt_push_train_aucs = []
gbt_push_val_aucs = []
gbt_push_train_ce = []
gbt_push_val_ce = []

fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(14,7))
fig2.suptitle("Validation metrics")

n_trees_grid = [5, 25, 50, 100]
for lr in [0.05, 0.1]:
    for depth in [1, 3, 5]:
        for n_trees in n_trees_grid:
            gbt = GradientBoostingClassifier(
                learning_rate=lr, max_depth=depth, n_estimators=n_trees
            )
            gbt.fit(X_train[train_important_cols_gbt], y_train)
            train_proba = gbt.predict_proba(X_train[train_important_cols_gbt])[:, 1]
            plot_metrics(f"gbt; LR = {lr}; MD = {depth}; n_trees = {n_trees}", y_pred = train_proba, y_test = train_feature_frame[label_col], figure = (fig1, ax1))

            val_proba = gbt.predict_proba(X_val[train_important_cols_gbt])[:, 1]
            plot_metrics(f"gbt; LR = {lr}; MD = {depth}; n_trees = {n_trees}", y_pred = val_proba, y_test = val_feature_frame[label_col], figure = (fig2, ax2))
plot_metrics(f"Baseline", y_pred=val_feature_frame['global_popularity'], y_test=val_feature_frame[label_col], figure=(fig2, ax2))
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_61_0.png)
    



    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_61_1.png)
    


### 4. Comparing models:


```python
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
```


```python
def plot_model_metrics(model_names, predictions, y_true):
    plt.figure(figsize=(14, 7))

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    for model_name, y_pred in zip(model_names, predictions):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision_score(y_true, y_pred):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # ROC Curve
    plt.subplot(1, 2, 2)
    for model_name, y_pred in zip(model_names, predictions):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.show()
```


```python
rf = RandomForestClassifier(100)
rf.fit(X_train[train_important_cols], y_train)
rf_predictions = rf.predict_proba(X_val[train_important_cols])[:, 1]
print("RF important columns")

rf_all = RandomForestClassifier(100)
rf_all.fit(X_train[train_cols], y_train)
rf_all_predictions = rf_all.predict_proba(X_val[train_cols])[:, 1]
print("RF all columns")

gbt = GradientBoostingClassifier(
        learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
)
gbt.fit(X_train[train_important_cols_gbt], y_train)
gbt_predictions = gbt.predict_proba(X_val[train_important_cols_gbt])[:, 1]
print("GBT important columns")

reduced_cols = ['ordered_before', 'abandoned_before', 'global_popularity']

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l1", C=1e-04, solver="saga")
)
lr.fit(X_train[reduced_cols], y_train)
lr_predictions = lr.predict_proba(X_val[reduced_cols])[:, 1]
print("Logistic regression")
```

    RF important columns
    RF all columns
    GBT important columns
    Logistic regression



```python
model_names = ["RF important columns", "RF all columns", "GBT important columns", "Logistic regression"]
predictions = [rf_predictions, rf_all_predictions, gbt_predictions, lr_predictions]

plot_model_metrics(model_names, predictions, y_val)
```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_66_0.png)
    


GBT is the best model for this scenario

### 5. Calibration:


```python
from sklearn.calibration import calibration_curve
```


```python
lr_prob_true, lr_prob_pred = calibration_curve(y_val, lr_predictions, n_bins=20)
gbt_prob_true, gbt_prob_pred = calibration_curve(y_val, gbt_predictions, n_bins=20)
```


```python
fig, ax= plt.subplots(figsize=(7, 7))

ax.plot(lr_prob_true, lr_prob_pred, label="logistic regression")
ax.plot(gbt_prob_true, gbt_prob_pred, label="gradient boosting trees")
ax.plot([0,1], [0,1], color="k", linestyle="--", label="perfect")
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1.05)
ax.legend()
```




    <matplotlib.legend.Legend at 0x7f34892d57d0>




    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_71_1.png)
    


GBT is better calibrated


```python
from sklearn.calibration import CalibratedClassifierCV
```


```python
calibrated_lr = CalibratedClassifierCV(lr, cv="prefit", method="isotonic")
calibrated_gbt = CalibratedClassifierCV(gbt, cv="prefit", method="isotonic")
```


```python
calibrated_lr.fit(X_val[reduced_cols], y_val)
val_lr_calibrated_pred = calibrated_lr.predict_proba(X_val[reduced_cols])[:, 1]
test_lr_calibrated_pred = calibrated_lr.predict_proba(X_test[reduced_cols])[:, 1]

calibrated_gbt.fit(X_val[train_important_cols_gbt], y_val)
val_gbt_calibrated_pred = calibrated_gbt.predict_proba(X_val[train_important_cols_gbt])[:, 1]
test_gbt_calibrated_pred = calibrated_gbt.predict_proba(X_test[train_important_cols_gbt])[:, 1]
```


```python
lr_prob_true_val_calibrated, lr_prob_pred_val_calibrated = calibration_curve(
    y_val, val_lr_calibrated_pred, n_bins=20
)
lr_prob_true_test_calibrated, lr_prob_pred_test_calibrated = calibration_curve(
    y_test, test_lr_calibrated_pred, n_bins=20
)

gbt_prob_true_val_calibrated, gbt_prob_pred_val_calibrated = calibration_curve(
    y_val, val_gbt_calibrated_pred, n_bins=20
)
gbt_prob_true_test_calibrated, gbt_prob_pred_test_calibrated = calibration_curve(
    y_test, test_gbt_calibrated_pred, n_bins=20
)
```


```python
fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(lr_prob_true_val_calibrated, lr_prob_pred_val_calibrated, label="Logistic Regression (Validation)", color="C8", linestyle="--")
ax.plot(gbt_prob_true_val_calibrated, gbt_prob_pred_val_calibrated, label="Gradient Boosting Tree (Validation)", color="C1", linestyle="--")
ax.plot([0, 1], [0, 1], color="k", linestyle="--", label="Perfect Calibration")

ax.plot(lr_prob_true_test_calibrated, lr_prob_pred_test_calibrated, label="Logistic Regression (Test)", color="C8")
ax.plot(gbt_prob_true_test_calibrated, gbt_prob_pred_test_calibrated, label="Gradient Boosting Tree (Test)", color="C1")

ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1.05)
ax.set_xlabel("True Probability")
ax.set_ylabel("Predicted Probability")
ax.legend()
plt.title("Calibration Plots")
plt.show()

```


    
![png](pull_notifications_ds_exploration_files/pull_notifications_ds_exploration_77_0.png)
    


GBT is still better calibrated besides not been proper calibrated at all.

### 6. Assesing final performance:


```python
from sklearn.metrics import precision_recall_fscore_support
```


```python
th = 0.05
gbt_test_binary_pred = (test_gbt_calibrated_pred > th).astype(int)
lr_test_binary_pred = (test_lr_calibrated_pred > th).astype(int)
```


```python
def get_model_binary_metrics(y_true, y_pred, name=""):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    results = pd.Series(
        {
            "precision": precision[1],
            "recall": recall[1],
            "f1": f1[1],
            "prevalence": support[1] / (support[0] + support[1])
        },
        name=name,
    )
    return results
```


```python
gbt_test_results = get_model_binary_metrics(y_test, gbt_test_binary_pred, name="gbt")
lr_test_results = get_model_binary_metrics(y_test, lr_test_binary_pred, name="lr")

test_results = pd.concat([gbt_test_results, lr_test_results], axis=1)
```


```python
test_results
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
      <th>gbt</th>
      <th>lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.155138</td>
      <td>0.145403</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.411262</td>
      <td>0.375295</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.225291</td>
      <td>0.209599</td>
    </tr>
    <tr>
      <th>prevalence</th>
      <td>0.012874</td>
      <td>0.012874</td>
    </tr>
  </tbody>
</table>
</div>


