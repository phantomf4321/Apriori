# Impoting necessary libraries
import numpy as np
import pandas as pd
from scipy.spatial import distance
import time
from apriori import GROUP
import math

class Engine:
    def __init__(self, group):
        print("Engine constructure is called!")
        self.group = group

    # to measure the elapsed wall-clock time between two points
    def time_measure(self, fuction):
        start = time.time()
        fuction
        end = time.time()
        print("your function is spending {}s".format(end - start))


    def mae(self, y_true, predictions):
        y_true, predictions = np.array(y_true), np.array(predictions)
        return np.mean(np.abs(y_true - predictions))

    def gsm(self):
        """
        Agha man digeh rad dadam...
        """




    def get_group(self):
        return self.group

    def monitor_group(self):
        print(self.get_group())

    def calculate_trust(self):
        members = self.group.index

        Trust_matrix = pd.DataFrame(0.0, index=members, columns=members)

        for u in members:
            rated_list_u = self.group.loc[u].index[Group.loc[u] > 0]
            count_rated_u = len(rated_list_u)
            ratings_u = Group.loc[u][:]

            for v in members:
                if u == v:
                    continue

                rated_list_v = Group.loc[v].index[Group.loc[v] > 0]
                ratings_v = Group.loc[v][:]

                intersection_uv = set(rated_list_u).intersection(rated_list_v)
                count_intersection = len(intersection_uv)

                partnership_uv = count_intersection / count_rated_u

                dst_uv = 1 / (1 + distance.euclidean(ratings_u, ratings_v))

                trust_uv = (2 * partnership_uv * dst_uv) / (partnership_uv + dst_uv)
                Trust_matrix.at[u, v] = trust_uv

        return Trust_matrix

    def calculate_similarity(self):
        members = self.group.index
        ratings = self.group.to_numpy()  # Convert DataFrame to a NumPy array

        # Calculate the Pearson correlation coefficient similarity
        PCC = np.corrcoef(ratings, rowvar=True)

        # Convert the matrix to a DataFrame with proper index and columns
        PCC_df = pd.DataFrame(PCC, index=members, columns=members)

        return PCC_df

    def calculate_centerality(self):
        members = self.group.index
        ratings = self.group.to_numpy()  # Convert DataFrame to a NumPy array
        matrix = np.zeros((len(members), len(members)))
        avg = np.average(ratings)

        loop_counter = 0
        for r in ratings:
            matrix[loop_counter][loop_counter] = abs(avg - np.average(r))
            loop_counter += 1

        # Convert the matrix to a DataFrame with proper index and columns
        Cd = pd.DataFrame(matrix, index=members, columns=members)

        return Cd


    def identify_leader(self, Trust_matrix, Similarity_matrix, Centerality_matrix, total_members):

        trust_sum = np.sum(Trust_matrix.values, axis=0) - 1
        similarity_sum = np.sum(Similarity_matrix.values, axis=0) - 1
        centerality_sum = np.sum(Centerality_matrix.values, axis=0) - 1

        ts_sumation = trust_sum + similarity_sum + centerality_sum

        LeaderId = np.argmax(ts_sumation)
        LeaderImpact = ts_sumation[LeaderId] / (total_members - 1)

        return Trust_matrix.index[LeaderId], LeaderImpact




    def identify_leader_without_centrality(self, Trust_matrix, Similarity_matrix, total_members):

        trust_sum = np.sum(Trust_matrix.values, axis=0) - 1
        similarity_sum = np.sum(Similarity_matrix.values, axis=0) - 1

        ts_sumation = trust_sum + similarity_sum

        LeaderId = np.argmax(ts_sumation)
        LeaderImpact = ts_sumation[LeaderId] / (total_members - 1)

        return Trust_matrix.index[LeaderId], LeaderImpact

    def calculate_influence_weight(self, leader_id, leader_impact, similarity_uv, trust_uv, v):
        if v == leader_id:
            weight_uv = (1 / 2) * ((leader_impact + (similarity_uv * trust_uv)) / (similarity_uv + trust_uv))
        else:
            weight_uv = (similarity_uv * trust_uv) / (similarity_uv + trust_uv)

        return weight_uv

    def influenced_rating(self, centerality):

        members = self.group.index
        movies = self.group.columns
        num_members, num_items = len(members), len(movies)

        # Calculate trust and similarity matrices
        trust_matrix = self.calculate_trust()
        similarity_matrix = self.calculate_similarity()
        centerality_matrix = self.calculate_centerality()

        # Identify the leader and their impact
        if(centerality):
            leader_id, leader_impact = self.identify_leader(trust_matrix, similarity_matrix, centerality_matrix, num_members)
        else:
            leader_id, leader_impact = self.identify_leader_without_centrality(trust_matrix, similarity_matrix, num_members)

        influenced_ratings = pd.DataFrame(0.0, index=members, columns=movies)

        for u in members:
            for i in movies:
                score_ui = self.group.at[u, i]
                influence = 0

                if score_ui > 0:
                    for v in members:
                        if v != u:
                            score_vi = self.group.at[v, i]
                            similarity_uv = similarity_matrix.at[u, v]
                            trust_uv = trust_matrix.at[u, v]
                            weight_vu = self.calculate_influence_weight(leader_id, leader_impact, similarity_uv, trust_uv, v)

                            if score_vi > 0:
                                influence += weight_vu * (score_vi - score_ui)

                    influenced_ratings.at[u, i] = score_ui + influence

        return influenced_ratings

    def evaluate_recommendations(self, Group_Rating, rec_size, satisfied_Tr):

        predict_list = []

        Group_Rating = Group_Rating.sort_values(ascending=False)
        rec_list = Group_Rating[Group_Rating != 0]

        recommendation_index = rec_list.index
        members = self.group.index
        no_member = len(members)

        TP = TN = FP = FN = 0
        satisfied = 0

        for r, index in enumerate(recommendation_index):
            for u in members:
                preference_u_ind = Group.at[u, index]

                if r < rec_size:
                    if preference_u_ind >= satisfied_Tr:
                        satisfied += 1
                        TP += 1
                        predict_list.append(1)
                    else:
                        FP += 1
                        predict_list.append(0)
                else:
                    if preference_u_ind >= satisfied_Tr:
                        FN += 1
                        predict_list.append(0)
                    else:
                        TN += 1
                        predict_list.append(1)

        total_count = TP + FP + TN + FN

        accuracy = ((TP + TN) / total_count) * 100 if total_count > 0 else 0
        precision = (TP / (TP + FP)) * 100 if TP + FP > 0 else 0
        recall = (TP / (TP + FN)) * 100 if TP + FN > 0 else 0
        specificity = (TN / (TN + FP)) * 100 if TN + FP > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        balanced_accuracy = (specificity + recall) / 2

        true_list = np.ones(len(predict_list))
        predict_list = np.array(predict_list)

        MAE = self.mae(true_list, predict_list)

        MSE = np.square(np.subtract(true_list, predict_list)).mean()
        RMSE = math.sqrt(MSE)

        GMS = self.gsm()

        results = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Balanced_Accuracy": balanced_accuracy,
            "MAE": MAE,
            "RMSE": RMSE,
            "GMS": GMS,
            "Confusion_counters": {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
        }

        return results

    def run(self, centrality):

        # Calculate members' influenced ratings
        Influenced_Ratings = self.influenced_rating(centrality)

        # Determine group rating for items using averaging aggregation method
        Group_Rating = Influenced_Ratings.mean(axis=0).fillna(0)

        # Evaluate the recommendations
        rec_size = 5
        satisfied_Tr = 4
        Evaluation_Results = self.evaluate_recommendations(Group_Rating, rec_size, satisfied_Tr)

        print("Evaluation Results:", Evaluation_Results)

        # Save the evaluation results to an Excel file
        results_df = pd.DataFrame(Evaluation_Results, index=["Evaluation Results"])
        results_df.to_excel('Results/Evaluation_Results.xlsx')


    def run_val(self, centrality):

        # Calculate members' influenced ratings
        Influenced_Ratings = self.influenced_rating(centrality)

        # Determine group rating for items using averaging aggregation method
        Group_Rating = Influenced_Ratings.mean(axis=0).fillna(0)

        # Evaluate the recommendations
        rec_size = 5
        satisfied_Tr = 4
        Evaluation_Results = self.evaluate_recommendations(Group_Rating, rec_size, satisfied_Tr)

        return Evaluation_Results
