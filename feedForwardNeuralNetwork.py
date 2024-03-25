import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
seed = 1234
np.random.seed(seed)

#DATA
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

#print(test_data.head())
#print(train_data.head())

#categorical features handling
categorical_cols = ['race', 'gender', 'age', 'diabetesMed', 'readmitted']

train_encode = pd.get_dummies(train_data, columns= categorical_cols)
test_encode = pd.get_dummies(test_data, columns= categorical_cols)

#create feature matrix and target
target_col = ['readmitted_<30','readmitted_>30','readmitted_NO']

X_tr = train_encode.drop(columns= target_col)
y_tr = train_encode[target_col].idxmax(axis= 1)
X_te = test_encode.drop(columns= target_col)
y_te = test_encode[target_col].idxmax(axis= 1) #combine one-hot encodings

#print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

#scale data to help with convergence
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr) #Fit and transform
X_te_scaled = scaler.transform(X_te)
print(X_tr_scaled.shape, y_tr.shape, X_te_scaled.shape, y_te.shape)

#Create Validation set
X_train, X_val, y_train, y_val = train_test_split(X_tr_scaled, y_tr, test_size=0.25, random_state=seed)

print("Feed Forward Neural Network")


#Using grid search

parameter_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 100), (150, 150)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'batch_size': [32, 64, 128, 256],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [100, 200, 300]
}

#Grid search or not
grid = True

if grid:   
    MLP = MLPClassifier(random_state= seed)
    grid_search = GridSearchCV(MLP, parameter_grid, cv= 5, scoring= 'accuracy')
    grid_search.fit(X_tr_scaled, y_tr)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_MLP = grid_search.best_estimator_
    MLP_prediction = best_MLP.predict(X_te_scaled)
    print(f'Accuracy score: {accuracy_score(y_te, MLP_prediction)}')
    
    train_error = 1 - best_MLP.score(X_tr_scaled, y_tr)
    test_error = 1 - best_MLP.score(X_te_scaled, y_te)
    print(f"Training error: {train_error}")
    print(f"Testing error: {test_error}")

if not grid:
    #hyper parameters
    hs = (144,144) #hidden_layer_sizes HISTORY: 100,100
    act = 'logistic' #activation HISTORY: relu
    sol = 'adam' #solver HISTORY: adam
    bs = 256 #batch_size
    lr = 0.001 #learning_rate
    nc = 100 #n_iter_no_change
    mi = 100 #max_iter
    
    MLP = MLPClassifier(hidden_layer_sizes= hs, activation= act, solver= sol, batch_size= bs, learning_rate= 'constant', learning_rate_init= lr, n_iter_no_change= nc, max_iter= mi, random_state= seed)
    MLP.fit(X_train, y_train)
    val_MLP_prediction = MLP.predict(X_val) #validation
    print(f'Validation Accuracy: {accuracy_score(y_val, val_MLP_prediction)}')
    
    test_MLP_prediction = MLP.predict(X_te_scaled) #test
    print(f'Test Accuracy: {accuracy_score(y_te, MLP_prediction)}')
    
    train_error = 1 - MLP.score(X_tr_scaled, y_tr)
    test_error = 1 - MLP.score(X_te_scaled, y_te)
    print(f"Training Error: {train_error}")
    print(f"Testing Error: {test_error}")


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #True

#Test if torch is working
x = torch.rand(5,3)
print(x)
#Test matplotlib
plt.plot([1, 2, 3, 4])
plt.ylabel('Some numbers')
plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        #create layers
        self.l1 = nn.Linear(input_size, hidden_size) #linear layer (input_size, output_size)
        self.relu = nn.ReLU()#activation function
        self.l2 = nn.Linear(hidden_size, num_classes) #linear layer

    def forward(self, x): #one sample x
        out = self.l1(x) #l1
        out = self.relu(out) #activation
        out = self.l2(out) #apply l2
        
        return out
model = NeuralNet(input_size, hidden_size, num_classes)

#loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

#training loop

#follow example in neural networks 2 slide
#relu
#softmax
'''