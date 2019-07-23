import pandas as pd




def preprocess(data, parameters):
    columns = parameters['columns']
    categ_columns = parameters['categ_columns']
    drop_columns = parameters["drop_columns"]

    if len(drop_columns) != 0:
        data.drop(columns=drop_columns, inplace=True)
    data.dropna(inplace=True)

    for column in categ_columns:
        x_cat = pd.get_dummies(data[column])
        data.drop(columns=column, inplace=True)
        data.join(x_cat)

