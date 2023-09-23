import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Data sets
movie_list=pd.read_csv('Movie Interests.csv')     # print(movie_list)
input_data=movie_list.drop(columns=['Interest'])  # print(input_data)
output_data=movie_list['Interest']                # print(output_data)

# Decision Tree from scikit learn module
movie_model=DecisionTreeClassifier()
movie_model.fit(input_data,output_data)    #match the input and output
movie_interest=movie_model.predict ([[9,1],[33,0]])
print(movie_interest)



