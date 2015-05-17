a=1
def add(x,y):
    return x+y

function_param={'x' :1,'y':3}
print(add(**function_param))