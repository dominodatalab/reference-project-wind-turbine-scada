import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

def data_prep(dataDF):
    
    dataDF["Date/Time"] = pd.to_datetime(dataDF["Date/Time"])
    dataDF["Month"] = pd.DatetimeIndex(dataDF["Date/Time"]).month
    dataDF["Hour"] = pd.DatetimeIndex(dataDF["Date/Time"]).hour
    dataDF.drop(["Date/Time"], axis=1, inplace=True)
    
    dataDF = dataDF.drop(dataDF[(dataDF["Month"]==1) | (dataDF["Month"]==12)].index)
    Q1 = dataDF["Wind Speed (m/s)"].quantile(0.25)
    Q3 = dataDF["Wind Speed (m/s)"].quantile(0.75)
    IQR = Q3 - Q1
    dataDF = dataDF[~((dataDF["Wind Speed (m/s)"] < (Q1 - 1.5 * IQR)) | (dataDF["Wind Speed (m/s)"] > (Q3 + 1.5 * IQR)))]
    dataDF = dataDF[~((dataDF["LV ActivePower (kW)"] == 0) & (dataDF["Theoretical_Power_Curve (KWh)"] !=0))]
    
    return dataDF

def train():
    print("Ingesting data...")
    dataDF = pd.read_csv("data/T1.csv")
    
    print("Preparing data...")
    dataDF = data_prep(dataDF)
    trainDF, testDF = train_test_split(dataDF, test_size=0.2, random_state=1234)
    
    X_train = trainDF.drop(columns=["LV ActivePower (kW)"]).values
    y_train = trainDF["LV ActivePower (kW)"].values
    X_test = testDF.drop(columns=["LV ActivePower (kW)"]).values
    y_test = testDF["LV ActivePower (kW)"].values

    print("Model training...")
    model = ExtraTreesRegressor(n_estimators=100, random_state=1234)
    model.fit(X_train, y_train)
    
    print("Model evaluation...")
    print("Accuracy on train: {:.0%}".format(model.score(X_train,y_train)))
    print("Accuracy on test : {:.0%}".format(model.score(X_test,y_test)))

    print("Persisting model...")
    
    with open("model.bin", 'wb') as f_out:
        pickle.dump(model, f_out)
        f_out.close()

    print("Model training completed.")
    
def main():
    train()

if __name__ == "__main__":
    main()