import pickle
    
with open("model.bin", "rb") as f_in:
    loaded_model = pickle.load(f_in)
    f_in.close()
    
def score(wind_speed, theoretical_power, wind_direction, month, hour):
    ret_val = loaded_model.predict([[wind_speed, theoretical_power, wind_direction, month, hour]])
    return ret_val[0]

if __name__ == "__main__":
    
    test_data = [12.70539, 3591.300461, 196.042206, 2, 0]
    
    print("Testing the model with test data :", test_data)
    print("Result : ", score(*test_data))
