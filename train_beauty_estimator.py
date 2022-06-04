import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import json
import pickle

## read training data
def read_data():
    full_data = []
    with open('./new_train.jsonl') as f:
        for line in f:
            full_data.append(json.loads(line))
    train_data = []
    train_scores = []
    for data in full_data:
        train_data.append(np.multiply(np.array(data['distances']), 1 / data['area_sqrt']))
        train_scores.append(data['score'])
    return train_data, train_scores


def svc_param_selection(X, y, nfolds, param_grid):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def main():
    train_data, train_scores = read_data()
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'epsilon': [1, 0.1, 0.01, 0.001]}
    result = svc_param_selection(train_data, train_scores, 5, param_grid)
    # train a regressor with the best parameter
    print(result)
    regressor = SVR(kernel = 'rbf', C=result['C'], gamma=result['gamma'], epsilon=result['epsilon'] )
    regressor.fit(train_data, train_scores)

    # save the model to disk
    filename = './beauty_score_model.sav'
    pickle.dump(regressor, open(filename, 'wb'))

if __name__ == "__main__":
    main()

## Usage: load the trained model
## loaded_model = pickle.load(open(filename, 'rb'))