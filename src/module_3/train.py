from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    from sklearn.model_selection import train_test_split
    order_ids = df[order_id_col].unique()
    
    order_train, order_test = train_test_split(order_ids, test_size=0.3, random_state=42)
    
    train_set = df[df[order_id_col].isin(order_train)]
    test_set = df[df[order_id_col].isin(order_test)]
    
    X_train = train_set.drop(columns=[outcome])
    y_train = train_set[[outcome]]
    
    X_test = test_set.drop(columns=[outcome])
    y_test = test_set[[outcome]]
    
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def build_feature_frame():
    return(
        load_dataset()
        .pipe(push_relevant_orders)
    )

def model_selection(df):
    import xgboost as xgb
    from itertools import product
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
    parameters = {'n_estimators': [25, 50, 75, 100], 'max_depth': [3, 5, 7, 9]}

    X_train, X_test, y_train, y_test = train_split_data(df, 'order_id', 'outcome')
    X_train_scaled, X_test_scaled=scale_data(X_train, X_test)


    # Crear una lista de todas las combinaciones de hiperparámetros
    param_combinations = list(product(parameters['n_estimators'], parameters['max_depth']))
    # Iterar sobre todas las combinaciones de hiperparámetros
    best_auc_pr=0
    for  (n_estimators, max_depth) in (param_combinations):
        # Crear y entrenar el modelo RandomForest con la combinación actual de hiperparámetros
        xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)

        
        # Calcular las predicciones para X_test
        y_pred_test= xgb_model.predict_proba(X_test_scaled)[:, 1]

        roc_auc_test = roc_auc_score(y_test, y_pred_test)
        
        precision_val, recall_val, _ = precision_recall_curve(y_test, y_pred_test)
        pr_auc_test = auc(recall_val, precision_val)

        if best_auc_pr<pr_auc_test:
            best_auc_pr=pr_auc_test
            best_estimators=n_estimators
            best_max_depth=max_depth
    logger.info(f"Best model is n_estimators={best_estimators} and max_depth={best_max_depth} with an AUCpr={best_auc_pr}")
    best_model= make_pipeline{
        StandardScaler(), xgb.XGBClassifier(n_estimators=best_estimators, max_depth=best_max_depth)
    }
    best_model.fit(X_train,y_train)

    save_model(best_model)







def main():
    feature_frame=build_feature_frame()
    model_selection(feature_frame)




if __name__== "__main__":
    main()