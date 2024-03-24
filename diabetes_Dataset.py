import pandas as pd
from pyxlsb import open_workbook
from sklearn.model_selection import train_test_split
import numpy as np

def read_xlsb(file_path):
    with open_workbook(file_path) as wb:
        with wb.get_sheet(1) as sheet:
            data = []
            for row in sheet.rows():
                data.append([cell.v for cell in row])
    df = pd.DataFrame(data[1:], columns=data[0]) #1st row column names
    return df

def get_Data():
    file = "C:\\Users\\hanjd\\Desktop\\178\\Project\\data.xlsb"
    df = read_xlsb(file)
    X = df[['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 'diabetesMed']]
    y = df['readmitted']
    
    #pandas dataframes
    return X, y

def split_Data():
    X,y = get_Data()
    #Split data into training(80%) and testing(20%)
    seed = 1234
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= seed)
    
    train_data = X_train.join(y_train)
    train_data.to_csv('train_data.csv', index = False)  # Save to CSV file without index

    test_data = X_test.join(y_test)
    test_data.to_csv('test_data.csv', index = False)  # Save to CSV file without index    

split_Data()



'''
from ucimlrepo import fetch_ucirepo 
# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
# data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
# metadata 
print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
# variable information 
print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 
'''
