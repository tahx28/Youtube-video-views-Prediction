import pandas as pd 

first_model = pd.read_csv('submission1.csv') # The path must be adapted
second_model = pd.read_csv('submission2.csv') # The path must be adapted

pred = first_model.copy()
pred['views'] = (first_model['views'] + second_model['views'] ) / 2

pred.to_csv('CombinedSubmission.csv', index=False)