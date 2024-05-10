from apriori import GROUP
from engine import Engine




G1 = GROUP('dataset/raiting.csv', 5)
Group = G1.set_matrix()
print(Group)

#Group.to_csv('dataset/my_group_on_5.csv')

# Assuming df is your DataFrame
columns_to_keep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_df = Group[columns_to_keep]
print(new_df)

import pandas as pd
import numpy as np


def calculate_similarity(group):
    members = group.index
    ratings = group.to_numpy()  # Convert DataFrame to a NumPy array

    # Calculate the Pearson correlation coefficient similarity
    curr_dataframe = np.corrcoef(ratings, rowvar=True)

    # Convert the matrix to a DataFrame with proper index and columns
    final_dataframe = pd.DataFrame(curr_dataframe, index=members, columns=members)

    return final_dataframe

# Create a dictionary with ten columns
data = {
    'I2': np.array([0, 0, 3.5, 0, 3]),
    'I3': np.array([0, 3.5, 0, 0, 4]),
    'I5': np.array([0, 3.5, 0, 0, 3]),
    'I6': np.array([5, 0, 0, 0, 0]),
    'I7': np.array([0, 3, 1, 0, 3.5]),
    'I8': np.array([0, 0, 3.5, 0, 0]),
    'I345': np.array([0, 3, 0, 2.5, 3]),
    'I370': np.array([0, 0, 0, 3, 0]),
    'I585': np.array([0, 3, 1, 3, 2.5]),
    'I638': np.array([0, 4, 4, 0, 3.5]),
}

# Create the DataFrame
df = pd.DataFrame(data)

print(df)
print(calculate_similarity(df))






"""
G1 = GROUP('dataset/raiting.csv', 5)
Group = G1.set_matrix()
Sub_group = G1.set_sub_matrix(10)

E1 = Engine(Group)
E2 = Engine(Sub_group)

print(E1.run(True))
print(E2.calculate_similarity())"""
