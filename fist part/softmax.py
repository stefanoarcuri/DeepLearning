import numpy as np
def softmax (L):
    expl=np.exp(L)
    sumExpl=sum(expl)
    result =[]
    for i in expl:
        result.append(i*1.0/sumExpl)
    print (result)
    pass
