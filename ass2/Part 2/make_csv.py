import pandas
from csv import reader

dataframe = pandas.read_csv("traindata.txt",delimiter="\t")
dataframe.to_csv("traindata.csv", encoding='utf-8', index=False)

dataframe = pandas.read_csv("testdata.txt",delimiter="\t")
dataframe.to_csv("testdata.csv", encoding='utf-8', index=False)
