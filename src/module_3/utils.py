path=('/home/raul/Documentos/zrive-ds/Datos/')

def load_dataset():
    dataset_name = "feature_frame.csv"
    loading_file= os.path.join(path, dataset_name)
    return pd.read_csv(loading_file)

def push_relevant_orders(df, min_products):
    basket_counts = df.groupby('order_id').outcome.sum()
    basket_of_min_size = basket_counts[basket_counts >= min_products].index
    return df.loc[lambda x: x.basket_counts.isin(basket_of_min_size)]

def train_split_data(df, order_id_col='order_id', outcome='outcome'):
    
    order_ids = df[order_id_col].unique()
    
    order_train, order_test = train_test_split(order_ids, test_size=0.3, random_state=42)
    
    train_set = df[df[order_id_col].isin(order_train)]
    test_set = df[df[order_id_col].isin(order_test)]
    
    X_train = train_set.drop(columns=[outcome])
    y_train = train_set[[outcome]]
    
    X_val = val_set.drop(columns=[outcome])
    y_val = val_set[[outcome]]
    
    X_test = test_set.drop(columns=[outcome])
    y_test = test_set[[outcome]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_feature_frame():
    return(
        load_dataset()
        .pipe(push_relevant_orders)
    )