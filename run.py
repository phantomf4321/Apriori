from apriori import GROUP
from engine import Engine

G1 = GROUP('dataset/raiting.csv', 5)
Group = G1.set_matrix()
Sub_group = G1.set_sub_matrix(10)
E1 = Engine(Group)
E2 = Engine(Sub_group)

print(E1.run(True))
print(E2.calculate_similarity())
