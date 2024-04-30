# -*- coding: utf-8 -*-
# @Time    : 2024/4/29


import pandas as pd

def get_data():
    df = pd.DataFrame(columns=['label', 'text'])
    
    with open('./data/toutiao_cat_data/toutiao_cat_data.txt', 'r', encoding='utf-8') as file:
        for line in file:
            label, text = create_dataset(line)
            print(label)
            df = df._append({'label': label, 'text': text}, ignore_index=True)
    df.to_csv('./data/toutiao_cat_data/toutiao_cat_data.csv', index=False, header=True)

def create_dataset(data):
        items = data.split('_!_')
        code = items[2]
        title = items[3]
        keyword = items[4]
        label = code
        text = title + keyword
        return label, text

if __name__ == "__main__":
    get_data()
