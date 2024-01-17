from apriori import GROUP
from engine import Engine

G1 = GROUP('dataset/raiting.csv', 10)
Group = G1.set_matrix()
E1 = Engine(Group)

print(E1.run(True))
