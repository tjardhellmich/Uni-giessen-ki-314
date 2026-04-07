import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

df["Price"] = housing.target

X = df.drop("Price", axis=1)
y = df["Price"]

from sklearn.model_selection import train_test_split

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_tf = scaler.fit_transform(X_train_tf)
X_test_tf = scaler.transform(X_test_tf)

param_df = pd.read_csv("/Users/oli/Desktop/Testfile_NEU.csv", sep=";", decimal=",")

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []
start = 1185
c = 0

for index, row in param_df.iterrows():

    if c < start:
        c = c + 1
        continue
    
    try:
        index_nr = row["Nr."]
        act_func = row["Activation Function"]
        layers = row["Anzahl Layer"]
        optimizer = row["Optimizer"]
        l_rate = row["Learning Rate"]
        loss_function = row["Loss Function"]
        #epochs = row["Epochs"]
        batch_size = row["Batch Size"]
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        if optimizer.lower() == "adam":
            ker_opt = keras.optimizers.Adam(learning_rate=l_rate)
        elif optimizer.lower() == "rmsprop":
            ker_opt = keras.optimizers.RMSprop(learning_rate=l_rate)
        elif optimizer.lower() == "sgd":
            ker_opt = keras.optimizers.SGD(learning_rate=l_rate)

        tf_model = tf.keras.Sequential()
        
        #Input Layer
        tf_model.add(keras.layers.Input(shape=(X_train_tf.shape[1],)))
        
        #Amount of layers
        for i in range(layers - 1):
            #Leaky ReLu muss anders angewendet werden als die übrigen Activation Functions
            if act_func == "Leaky ReLU":
                tf_model.add(keras.layers.Dense(64))
                tf_model.add(keras.layers.LeakyReLU())
            else:
                tf_model.add(keras.layers.Dense(64, activation=act_func))
        
        #Output Layer
        tf_model.add(keras.layers.Dense(1))
        
        tf_model.compile(
            optimizer=ker_opt,
            loss=loss_function,
            metrics=['mae']
        )
        
        history = tf_model.fit(
            X_train_tf,
            y_train_tf,
            validation_split=0.2,
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Vorhersagen mit dem neuen Modell
        y_pred = tf_model.predict(X_test_tf).reshape(-1)
        
        # Metriken berechnen
        mae_test = mean_absolute_error(y_test_tf, y_pred)
        mse_test = mean_squared_error(y_test_tf, y_pred)
        r2_test = r2_score(y_test_tf, y_pred)

    except:
        mae_test = 0
        mse_test = 0
        r2_test = 0

    c = c + 1
    print("#########################")
    print(f"Kombination {c} fertig")
    print("#########################")
    print("Test Model 1 - MAE:", mae_test)
    print("Test Model 1 - MSE:", mse_test)
    print("Test Model 1 - R2:", r2_test)

    results.append({
        "Nr.": index_nr,
        "MAE": mae_test,
        "MSE": mse_test,
        "R2": r2_test
    })


    if c % 5 == 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv("/Users/oli/Desktop/test_results_NEU.csv", mode="a", index=False)
        results.clear()
        

results_df = pd.DataFrame(results)

results_df.to_csv("/Users/oli/Desktop/test_results_NEU.csv", mode="a", index=False)