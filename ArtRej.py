import numpy as np

def artifact_rejection(data):
    """
    Input:
        4*7680 observation data
    Output:
        4*None artifact removed observation data
    """
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    thre_h = 500
    #thre_l = 1
    for j in range(data.shape[1]):
        if data[0,j] > thre_h or data[0,j] < -thre_h:
            c1.append(j)

    for j in range(data.shape[1]):
        if data[1,j] > thre_h or data[1,j] < -thre_h:
            c2.append(j)

    for j in range(data.shape[1]):
        if data[2,j] > thre_h or data[2,j] < -thre_h:
            c3.append(j)

    for j in range(data.shape[1]):
        if data[3,j] > thre_h or data[3,j] < -thre_h:
            c4.append(j)

    # for j in range(data.shape[1]):
    #     if data[0,j] > thre_h or data[0,j] < -thre_h or (data[0,j]>(-thre_l) and data[0,j]<thre_l):
    #         c1.append(j)
    #
    # for j in range(data.shape[1]):
    #     if data[1,j] > thre_h or data[1,j] < -thre_h or (data[1,j]>(-thre_l) and data[1,j]<thre_l):
    #         c2.append(j)
    #
    # for j in range(data.shape[1]):
    #     if data[2,j] > thre_h or data[2,j] < -thre_h or (data[2,j]>(-thre_l) and data[2,j]<thre_l):
    #         c3.append(j)
    #
    # for j in range(data.shape[1]):
    #     if data[3,j] > thre_h or data[3,j] < -thre_h or (data[3,j]>(-thre_l) and data[3,j]<thre_l):
    #         c4.append(j)
    # s1 = set(c1)
    # s2 = set(c2)
    # s3 = set(c3)
    # s4 = set(c4)
    #
    # union = s1 | s2 | s3 | s4
    indice = np.union1d(c1,c2)
    indice = np.union1d(indice,c3)
    indice = np.union1d(indice,c4)

    newdata = []
    for i in range(data.shape[0]):
        newdata.append(np.delete(data[i],indice))
    newdata = np.array(newdata)
    return newdata