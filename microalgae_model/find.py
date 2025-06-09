import numpy as np
# 该函数用于将序列t划分为小于tf和大于tf的部分
# 默认t已经从小到大顺序排列，且无重复值
def Find(tf,t):
    t_low = []
    for i,v in enumerate(t):
        if v < tf:
            t_low.append(v)
        else:
            return [[np.array(t_low),i],t[i:]]
