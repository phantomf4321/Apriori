from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from collections import OrderedDict
import time


class GROUP:
    def __init__(self, dataset_path, number_of_elements):
        # Flag
        print("Group constructor is called to export {} elements from {}".format(number_of_elements, dataset_path))
        # Reading Dataset
        self.data_frame = pd.read_csv(dataset_path)
        self.number_of_elements = number_of_elements

    # to measure the elapsed wall-clock time between two points
    def time_measure(self, fuction):
        start = time.time()
        fuction
        end = time.time()
        print("your function is spending {}s".format(end - start))

    # Get header of Dataset
    def get_head(self):
        return self.data_frame.head()

    def get_user_id_cul(self):
        return self.data_frame['user_id']

    def get_item_id_cul(self):
        return self.data_frame['item_id']

    def get_raiting_value_cul(self):
        return self.data_frame['rating_value']

    def get_user_triple_id_cul(self):
        return self.data_frame[['user_id', 'user_id', 'user_id']]

    # Convert Dataframe to 2D list
    def local_convert_df_to_list(self):
        return self.data_frame.values.tolist()

    # Convert Dataframe to 2D list
    def convert_df_to_list(self, df):
        return df.values.tolist()

    # Kth maximume element:
    def Kth_max(self, index, input_list):
        input_list = sorted(input_list, reverse=True)
        return input_list[index - 1]

    # Apriori algrithem
    def Apriori(self, min_sup, curr_df):
        # Convert data to list
        curr_list = self.convert_df_to_list(curr_df)

        # Convert data to apropriate form of Aprioori Function
        te = TransactionEncoder()
        te_ary = te.fit_transform(curr_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Call apriori
        frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)

        return frequent_itemsets

    # Creating group:
    def group_creation(self):
        # initial state
        curr_sup = 0.5
        curr_number_of_elements = 0

        # O(2^k * n) for Apriori
        # O(log n for loop)
        # total time complexity : O(2^k * n * log n)
        while curr_number_of_elements < self.number_of_elements:
            curr_output = self.Apriori(curr_sup, self.get_user_triple_id_cul())
            if curr_output['support'].count() == self.number_of_elements:
                return curr_output
            else:
                curr_number_of_elements = curr_output['support'].count()
                curr_sup = curr_sup / 2
                loop_output = curr_output

        cur_list = curr_output['support'].values.tolist()
        kth_max = self.Kth_max(self.number_of_elements, cur_list)
        curr_output = self.Apriori(kth_max, self.get_user_triple_id_cul())

        return curr_output



    def group_elements(self):
        group = self.group_creation()
        group = group['itemsets'].values.tolist()
        for g in group:
            group[group.index(g)] = list(g)[0]

        return(group)


    def set_matrix(self):

        group_elements = self.group_elements()
        sub_list1 = self.data_frame['item_id'].values.tolist()
        list_of_cul = sub_list1


        list_of_cul.sort()
        list_of_cul = list(OrderedDict.fromkeys(list_of_cul))

        cur_matrix = []
        for ge in group_elements:
            curr_data = self.data_frame.loc[self.data_frame['user_id'] == ge]

            cur_row = []
            for loc in list_of_cul:
                cur_list = curr_data['item_id'].values.tolist()
                if loc in cur_list:
                    curr_rate = self.data_frame.loc[self.data_frame['user_id'] == ge]
                    curr_rate2 = curr_rate.loc[self.data_frame['item_id'] == loc]
                    rate_list = curr_rate2['rating_value'].values.tolist()
                    cur_row.append(rate_list[0])
                else:
                    cur_row.append(0)
            cur_matrix.append(cur_row)

        df = pd.DataFrame(cur_matrix, columns=list_of_cul)
        df.insert(0, "Unnamed: 0", group_elements, True)
        df.set_index("Unnamed: 0", inplace = True)
        return df


    def set_sub_matrix(self, number_of_movies):

        group_elements = self.group_elements()
        sub_list1 = self.data_frame['item_id'][0:number_of_movies].values.tolist()
        list_of_cul = sub_list1


        list_of_cul.sort()
        list_of_cul = list(OrderedDict.fromkeys(list_of_cul))

        cur_matrix = []
        for ge in group_elements:
            curr_data = self.data_frame.loc[self.data_frame['user_id'] == ge]

            cur_row = []
            for loc in list_of_cul:
                cur_list = curr_data['item_id'].values.tolist()
                if loc in cur_list:
                    curr_rate = self.data_frame.loc[self.data_frame['user_id'] == ge]
                    curr_rate2 = curr_rate.loc[self.data_frame['item_id'] == loc]
                    rate_list = curr_rate2['rating_value'].values.tolist()
                    cur_row.append(rate_list[0])
                else:
                    cur_row.append(0)
            cur_matrix.append(cur_row)

        df = pd.DataFrame(cur_matrix, columns=list_of_cul)
        df.insert(0, "Unnamed: 0", group_elements, True)
        df.set_index("Unnamed: 0", inplace = True)
        return df
