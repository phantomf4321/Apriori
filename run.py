from apriori import GROUP
from engine import Engine




"""G1 = GROUP('dataset/raiting.csv', 5)
Group = G1.set_matrix()
print(Group)

Group.to_csv('dataset/my_group_on_5.csv')

# Assuming df is your DataFrame
columns_to_keep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_df = Group[columns_to_keep]
print(new_df)"""

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


from scipy.spatial import distance


def calculate_trust(Group):
    """
    Calculate the trust matrix based on common rated items between group members and their ratings distance.

    Parameters:
        Group (DataFrame): A DataFrame containing group members' ratings for items.

    Returns:
        Trust_matrix (DataFrame): The trust matrix representing the trust levels between members.
    """
    members = Group.index
    no_member = len(members)

    Trust_matrix = pd.DataFrame(0.0, index=members, columns=members)

    for u in members:
        rated_list_u = Group.loc[u].index[Group.loc[u] > 0]
        count_rated_u = len(rated_list_u)
        ratings_u = Group.loc[u][:]

        for v in members:
            if u == v:
                continue

            rated_list_v = Group.loc[v].index[Group.loc[v] > 0]
            count_rated_v = len(rated_list_v)
            ratings_v = Group.loc[v][:]

            intersection_uv = set(rated_list_u).intersection(rated_list_v)
            count_intersection = len(intersection_uv)

            partnership_uv = count_intersection / count_rated_u

            dst_uv = 1 / (1 + distance.euclidean(ratings_u, ratings_v))

            trust_uv = (2 * partnership_uv * dst_uv) / (partnership_uv + dst_uv)
            Trust_matrix.at[u, v] = trust_uv

    return Trust_matrix



def calculate_centerality(group):
    members = group.index
    ratings = group.to_numpy()  # Convert DataFrame to a NumPy array
    matrix = np.zeros((len(members), len(members)))
    avg = np.average(ratings,  weights=(ratings != 0))

    loop_counter = 0
    for r in ratings:
        matrix[loop_counter][loop_counter] = abs(avg - np.average(r, weights=(r != 0)))
        loop_counter += 1

    # Convert the matrix to a DataFrame with proper index and columns
    Cd = pd.DataFrame(matrix, index=members, columns=members)

    return Cd


def calculate_centerality_list(group):
    resault = []
    members = group.index
    ratings = group.to_numpy()  # Convert DataFrame to a NumPy array
    avg = np.average(ratings,  weights=(ratings != 0))

    for r in ratings:
        resault.append(abs(avg - np.average(r, weights=(r != 0))))

    return resault

# Create a dictionary with ten columns
data = {
    'I1': np.array([2.5, 2, 2, 2, 2.5]),
    'I2': np.array([3, 3.5, 3.5, 3.5, 3]),
    'I3': np.array([3.5, 3.5, 3, 4, 4]),
    'I4': np.array([3.5, 0, 3.5, 3, 1]),
    'I5': np.array([2.5, 3.5, 0, 2, 3]),
    'I6': np.array([5, 4, 4.5, 3, 4.5]),
    'I7': np.array([2.5, 3, 3.5, 2.5, 3.5]),
    'I8': np.array([0, 3, 3.5, 1.5, 0]),
    'I9': np.array([0, 0, 2.5, 2, 2.5]),
    'I10': np.array([3.5, 3, 3, 3, 3.5]),
}


# Create the DataFrame
df = pd.DataFrame(data)

print("Dataframe\n", df)

print(len(df))

print(header)


"""
print("similarity:\n", calculate_similarity(df))
print("trust:\n", calculate_trust(df))
print("Centrality: \n", calculate_centerality(df))
print("Centerality list :", calculate_centerality_list(df))"""

"""trust = calculate_trust(df)
similarity = calculate_similarity(df)
df.to_csv('dataset/my_sample_dataframe.csv')
trust.to_csv('dataset/my_sample_dataframe_trust.csv')
similarity.to_csv('dataset/my_sample_dataframe_similarity.csv')"""





"""G1 = GROUP('dataset/raiting.csv', 5)
Group = G1.set_matrix()
Sub_group = G1.set_sub_matrix(10)

E1 = Engine(Group)
E2 = Engine(Sub_group)

print(E1.run(True))
print(E2.calculate_trust())
print(E2.calculate_similarity())"""
