def solve(a, b, c):
    resalts = []
    d = a + b + c
    resalts.append(d)
    d = a * b + c
    resalts.append(d)
    d = a * (b + c)
    resalts.append(d)
    d = a + b * c
    resalts.append(d)
    d = (a + b) * c
    resalts.append(d)
    d = a * b * c
    resalts.append(d)

    return max(resalts)

a = int(input())
b = int(input())
c = int(input())

print(solve(a, b, c))