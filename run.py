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
    d = a + b + c
    resalts.append(d)

    return max(resalts)

