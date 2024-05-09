from apriori import GROUP
from engine import Engine

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
    'I1': np.array([0, 3, 4, 3, 0]),
    'I2': np.array([2, 2, 0, 4, 2]),
    'I3': np.array([4, 0, 3, 0, 0]),
    'I4': np.array([3, 4, 5, 5, 3]),
    'I5': np.array([0, 0, 3, 2, 0]),
    'I6': np.array([5, 4, 4, 4, 5]),
    'I7': np.array([0, 2, 3, 0, 4]),

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
