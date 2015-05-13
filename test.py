def add1(x, y):
    return x + y


def add2(x, y):
    return x * y


def add3(x, y):
    return x / y


func = [add1, add2, add3][1]

print(func(2, 8))
