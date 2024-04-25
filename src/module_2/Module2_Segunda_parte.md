```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```


```python
df_feature=pd.read_csv('/home/raulherrero/datos-zrive/feature_frame.csv')
```


```python
df_feature.head()
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
df_feature.info()
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
info_cols= ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
label_col='outcome'
features_cols=[col for col in df_feature.columns if col not in info_cols + [label_col]]

categoriacal_cols=['product_type', 'vendor']
binary_cols=['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols= [col for col in features_cols if col not in categoriacal_cols + binary_cols]
```


```python
df_feature[label_col].value_counts()
```




    outcome
    0.0        2847317
    1.0          33232
    Name: count, dtype: int64




```python
for col in binary_cols:
    print(f"Value counts {col}: {df_feature[col].value_counts().to_dict()}")
    print(f"Mean outcome by {col} value: {df_feature.groupby(col)['outcome'].mean().to_dict()}")
    print("---")
```

    Value counts ordered_before: {0.0: 2819658, 1.0: 60891}
    Mean outcome by ordered_before value: {0.0: 0.008223337723936732, 1.0: 0.1649669080816541}
    ---
    Value counts abandoned_before: {0.0: 2878794, 1.0: 1755}
    Mean outcome by abandoned_before value: {0.0: 0.011106039542947498, 1.0: 0.717948717948718}
    ---
    Value counts active_snoozed: {0.0: 2873952, 1.0: 6597}
    Mean outcome by active_snoozed value: {0.0: 0.011302554809544488, 1.0: 0.1135364559648325}
    ---
    Value counts set_as_regular: {0.0: 2870093, 1.0: 10456}
    Mean outcome by set_as_regular value: {0.0: 0.010668992259135854, 1.0: 0.24971308339709258}
    ---



```python
corr=df_feature[numerical_cols + [label_col]].corr()

mask=np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11,9))

cmap= sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
```




    <Axes: >




    
![png](Module2_Segunda_parte_files/Module2_Segunda_parte_7_1.png)
    



```python
cols=3
rows= int(np.ceil(len(numerical_cols)/cols))
fig, ax = plt.subplots(rows, cols, figsize=(20,5 * rows))
ax=ax.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df_feature.loc[lambda x: x.outcome == 0, col], label='0', ax=ax[i])
    sns.kdeplot(df_feature.loc[lambda x: x.outcome == 1, col], label='1', ax=ax[i])
    ax[i].set_title(col)

ax[0].legend()
plt.tight_layout()

```


    
![png](Module2_Segunda_parte_files/Module2_Segunda_parte_8_0.png)
    



```python
print(df_feature['outcome'].unique())
print(df_feature['product_type'].unique())
print(df_feature['vendor'].unique())
```

    [0 1]
    ['ricepastapulses' 'snacksconfectionery' 'dishwasherdetergent'
     'cleaningaccessories' 'fabricconditionerfreshener' 'coffee'
     'femininecare' 'bathroomlimescalecleaner' 'handsoapsanitisers'
     'tinspackagedfoods' 'toiletroll' 'kitchenrolltissues' 'binbags'
     'windowglasscleaner' 'homebaking' 'tea' 'jamhoneyspreads'
     'washingliquidgel' 'longlifemilksubstitutes' 'allpurposecleaner'
     'softdrinksmixers' 'condimentsdressings' 'babyfood6months' 'kidssnacks'
     'cookingingredientsoils' 'floorcleanerpolish' 'cereal'
     'driedfruitsnutsseeds' 'pickledfoodolives' 'catfood'
     'cookingsaucesmarinades' 'juicesquash' 'beer' 'kidsdental' 'nappies'
     'maternity' 'washingpowder' 'dental' 'haircare' 'bathshowergel'
     'kitchenovencleaner' 'wipescottonwool' 'dogfood' 'babytoiletries'
     'foodstorage' 'shavinggrooming' 'petcare' 'deodorant' 'washingcapsules'
     'bodyskincare' 'delicatesstainremover' 'babyfood12months'
     'facialskincare' 'superfoodssupplements' 'dryingironing'
     'premixedcocktails' 'householdsundries' 'feedingweaning'
     'babymilkformula' 'nappypants' 'healthcarevitamins' 'airfreshener']
    ['clearspring' 'sparesnacks' 'finish' 'propercorn' 'ecoforce' 'ecover'
     'cafédirect' 'totm' 'cif' 'marigold' 'faithinnature' 'difatti' 'andrex'
     'thecheekypanda' 'symphony' 'kleenex' 'cooksco' 'droetker' 'pukka'
     'stdalfour' 'oatly' 'tonyschocolonely' 'minorfigures' 'dashwater'
     'hellmanns' 'method' 'heinz' 'fish4ever' 'organix' 'biona' 'flash'
     'seventhgeneration' 'mutti' 'ecozone' 'dorsetcereals' 'pipnut' 'nairns'
     'nutella' 'treeoflife' 'drogheriaalimentari' 'matthewscotswoldflour'
     'whogivesacrap' 'branston' 'fevertree' 'felix' 'tatelyle' 'princes'
     'cocacola' 'robinsons' 'crespo' 'wholeearth' 'brewdog' '7up' 'bluedragon'
     'filippoberio' 'meridian' 'jacknjill' 'weetabix' 'kitkin' 'weleda'
     'ambrosia' 'sensodyne' 'carex' 'kallo' 'aspall' 'leaperrins'
     'simplygentle' 'taylorsofharrogate' 'mrmuscle' 'humblebrush' 'biod'
     'yarrah' 'wilkinsons' 'hp' 'plenty' 'maldon' 'jordans' 'flashpgp'
     'huggies' 'ecloth' 'stardrops' 'bisto' 'johnsons' 'persil' 'sealapack'
     'dove' 'gillette' 'catsan' 'sanex' 'mcvities' 'ryvita' 'knorr' 'kelloggs'
     'waterwipes' 'ariel' 'fourpurebrewingcompany' 'colmans' 'allinsons'
     'calgon' 'duck' 'twinings' 'clipper' 'tilda' 'spontex' 'minky' 'bother'
     'maryberrys' 'heineken' 'merchantgourmet' 'nandos' 'comfort' 'verylazy'
     'tropicalwholefoods' 'cawstonpress' 'linwoods' 'headshoulders' 'lenor'
     'eatnatural' 'alpro' 'pgtips' 'nescafé' 'oralb' 'fairy'
     'lovebeautyplanet' 'viakal' 'rudehealth' 'koko' 'rizopia' 'amadin'
     'dettol' 'paxo' 'kingfisher' 'colgate' 'ribena' 'dovesfarm' 'delmonte'
     'buitoni' 'ritz' 'rowse' 'panteneprov' 'radox' 'crazyjack' 'vegó'
     'tresemme' 'frenchs' 'braggs' 'pedigree' 'vaseline' 'cadbury' 'natracare'
     'brooklyn' 'robertsons' 'shreddedwheat' 'saltoftheearth'
     'quinolamothergrain' 'pukkateas' 'profusion' 'bakers' 'biobag' 'nestle'
     'encore' 'wingyip' 'carnation' 'flora' 'kühne' 'listerine' 'westlab'
     'piccolo' 'kingsoba' 'marmite' 'budweiserbudvar' 'karmadrinks'
     'bluediamond' 'originalsource' 'vitalbaby' 'febreze' 'ape' 'properchips'
     'zoflora' 'dualit' 'littlepastaorganics' 'whiskas' 'vanish'
     'herbalessences' 'batchelors' 'covermatebytouch' 'sunwheel'
     'thenaturalfamilyco' 'bart' 'kikkoman' 'harrybrand' 'karmacola'
     'ellaskitchen' 'always' 'nivea' 'kabutonoodles' 'artisangrains' 'bulldog'
     'ffs' 'pulsin' 'ifyoucare' 'cocofina' 'spare' 'quaker' 'bacofoil'
     'tabascooriginalred' 'drty' 'brita' 'shreddies' 'hartleys' 'milton'
     'freee' 'aptamil' 'baxters' 'nescafe' 'cowgate' 'nuk' 'schmidts'
     'hambledenhibiscustea' 'scotts' 'lilyskitchen' 'childsfarm' 'pampers'
     'bobsredmill' 'lynx' 'hollings' 'dreamies' 'fohwa' 'wellbaby' 'simple'
     'spam' 'maistic' 'greenpeople' 'sanchi' 'daioni' 'tampax' 'annabelkarmel'
     'betterbodyfoods' 'navitas' 'vitacoco' 'vicks' 'tommeetippee' 'carlsberg'
     'redbull' 'enterosgel' 'fody' 'equazen' 'lansinoh' 'munchkin' 'bimuno'
     'pantene' 'beamingbaby' 'silverspoon' 'cadburys' 'bamboonature' 'malibu'
     'vo5' 'provamel' 'pregnacare' 'hipporganic' 'loreal' 'mrsheen'
     'creativenature' 'colief']

