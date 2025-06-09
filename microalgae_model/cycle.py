import numpy as np
from .model3 import Logistic
from scipy.integrate import solve_ivp
from .find import Find

# class Modol(object):
#     def __init__(self, X0, I0, N0,  tf, T0,model,parameters_light=None,parameters_dark=None,ligh=True, L=0.06):
#         # 输入变量
#         X0 = X0
#         self.I0 = I0
#         N0 = N0
#         tf = tf
#         self.T0 = T0
#         self.model = model
#         self.parameter_light = parameters_light
#         self.parameter_dark = parameters_dark
#         self.ligh = ligh
#         self.L = L

            
# def Cycle(X0,N0,P0,model,tf,t_up,t_lb=0,num=None):
#     """
#     该函数用于模拟模型在t_lb到t_up时间之间微藻生长状态的动态过程
    
#     """
#     # 时间间隔默认为一小时
#     if num == None:
#         num = t_up - t_lb + 1
#     t = np.linspace(t_lb,t_up,num=num)
#     cycle = np.ceil(t[-1]/24)    # 光暗循环代数
#     for m in model:
#         if m.ligh == True:
#             model_light = m
#         if m.ligh == False:
#             model_dark = m
    
#     for i in range(int(cycle)):
#         j = i+1
#         if cycle == 0:
#             if t[-1] <= tf:
#                 X_ = solve_ivp(model_light.Simulation, [0,t[-1]], [X0,N0], t_eval=t)
#                 X = X_.y[0]
#                 N = X_.y[1]
#                 # L = X_.y[2]
#             else:
#                 X_1 = solve_ivp(model_light.Simulation, [0,tf], [X0,N0], t_eval=t[:tf])
#                 X_2 = solve_ivp(model_dark.Simulation,[tf,t[-1]],[X_1.y[0][-1],X_1.y[1][-1]],t_eval=t[tf:])
#                 X = np.append(X_1.y[0],X_2.y[0])
#                 N = np.append(X_1.y[1],X_2.y[1])
#                 # L = np.append(X_1.y[2],X_2.y[2])
#                 # X = X.append(X_2.y[0])
#                 # N = N.append(X_2.y[1])
#         elif i==0: #and t[-1] - 24*j >= 0:
#             X_1 = solve_ivp(model_light.Simulation, [t[0],t[0]+tf], [X0,N0], t_eval=t[:tf])
#             X_2 = solve_ivp(model_dark.Simulation, [t[0]+tf,t[0]+24*j], [X_1.y[0][-1],X_1.y[1][-1]], t_eval=t[tf:24])
#             X = np.append(X_1.y[0],X_2.y[0])
#             N = np.append(X_1.y[1],X_2.y[1])
#             # L = np.append(X_1.y[2],X_2.y[2])
#             # X = X.append(X_2.y[0])
#             # N = N.append(X_2.y[1])
#         elif i!=0 and t[-1] - 24*j >= 0:
#             X_1 = solve_ivp(model_light.Simulation, [t[0]+24*i,t[0]+24*i+tf], [X[-1],N[-1]], t_eval=t[24*i:24*i+tf])    
#             X_2 = solve_ivp(model_dark.Simulation, [t[0]+24*i+tf,t[0]+24*j], [X_1.y[0][-1],X_1.y[1][-1]], t_eval=t[24*i+tf:24*j])
#             X = np.concatenate((X,X_1.y[0],X_2.y[0]))
#             N = np.concatenate((N,X_1.y[1],X_2.y[1]))
#             # L = np.concatenate((L,X_1.y[2],X_2.y[2]))
#             # X = X.append(X_2.y[0])
#             # N = N.append(X_2.y[1])
#         elif i!=0 and t[-1] - 24*j < 0 and t[-1] - 24*i > tf:
#             X_1 = solve_ivp(model_light.Simulation, [t[0]+24*i,t[0]+24*i+tf], [X[-1],N[-1]], t_eval=t[24*i:24*i+tf])    
#             X_2 = solve_ivp(model_dark.Simulation, [t[0]+24*i+tf,t[-1]-24*i-tf], [X_1.y[0][-1],X_1.y[1][-1]], t_eval=t[24*i+tf:])
#             X = np.concatenate((X,X_1.y[0],X_2.y[0]))
#             N = np.concatenate((N,X_1.y[1],X_2.y[1]))
#             # L = np.concatenate((L,X_1.y[2],X_2.y[2]))
#             # X = X.append(X_2.y[0])
#             # N = N.append(X_2.y[1])
#         elif i!=0 and t[-1] - 24*j < 0 and t[-1] - 24*i <= tf:
#             X_ = solve_ivp(model_light.Simulation, [t[0]+24*i,t[-1]], [X[-1],N[-1]], t_eval=t[24*i:])
#             X = np.append(X,X_.y[0])
#             N = np.append(N,X_.y[1])
#             # L = np.append(L,X_.y[1])
    
#     return [X,N]
def Cycle(X0,N0,P0,model,tf,t_up,t_lb=0,num=None,t_exp=None,L0=None):
    """
    该函数用于模拟模型在t_lb到t_up时间之间微藻生长状态的动态过程
    
    """
    if t_exp is None:
        # 时间间隔默认为一小时
        if num == None:
            num = int(t_up - t_lb + 1)
        t = np.linspace(t_lb,t_up,num=num)
        cycle = np.ceil(t[-1]/24)    # 光暗循环代数
        for m in model:
            if m.ligh == True:
                model_light = m
            if m.ligh == False:
                model_dark = m
        for i in range(int(cycle)):
            j = i+1
            if cycle == 1:
                if t[-1] <= tf:
                    X_ = solve_ivp(model_light.Simulation, [0,t[-1]], [X0,N0,P0,L0], t_eval=t)
                    X = X_.y[0]
                    N = X_.y[1]
                    P = X_.y[2]
                    L = X_.y[3]
                    # L = X_.y[2]
                else:
                    X_1 = solve_ivp(model_light.Simulation, [0,tf], [X0,N0,P0,L0], t_eval=t[:tf])
                    # model_dark.X0=X_1.y[0][-1]
                    # model_dark.N0=X_1.y[1][-1]
                    # model_dark.P0=X_1.y[2][-1]

                    X_2 = solve_ivp(model_dark.Simulation,[tf,t[-1]],[X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]],t_eval=t[tf:])

                    X = np.append(X_1.y[0],X_2.y[0])
                    N = np.append(X_1.y[1],X_2.y[1])
                    P = np.append(X_1.y[2],X_2.y[2])
                    L = np.append(X_1.y[3],X_2.y[3])
                    # X = X.append(X_2.y[0])
                    # N = N.append(X_2.y[1])
            elif tf == 24:
                X_ = solve_ivp(model_light.Simulation, [0,t[-1]], [X0,N0,P0,L0], t_eval=t)
                X = X_.y[0]
                N = X_.y[1]
                P = X_.y[2]
                L = X_.y[3]
            elif i==0: #and t[-1] - 24*j >= 0:
                X_1 = solve_ivp(model_light.Simulation, [t[0],t[0]+tf], [X0,N0,P0,L0], t_eval=t[:tf])

                X_2 = solve_ivp(model_dark.Simulation, [t[0]+tf,t[0]+24*j], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t[tf:24])

                X = np.append(X_1.y[0],X_2.y[0])
                N = np.append(X_1.y[1],X_2.y[1])
                P = np.append(X_1.y[2],X_2.y[2])
                L = np.append(X_1.y[3],X_2.y[3])
                # X = X.append(X_2.y[0])
                # N = N.append(X_2.y[1])
            elif i!=0 and t[-1] - 24*j >= 0:
                X_1 = solve_ivp(model_light.Simulation, [t[0]+24*i,t[0]+24*i+tf], [X[-1],N[-1],P[-1],L[-1]], t_eval=t[24*i:24*i+tf])    
                X_2 = solve_ivp(model_dark.Simulation, [t[0]+24*i+tf,t[0]+24*j], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t[24*i+tf:24*j])
                X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                L = np.concatenate((L,X_1.y[3],X_2.y[3]))
                # X = X.append(X_2.y[0])
                # N = N.append(X_2.y[1])
            elif i!=0 and t[-1] - 24*j < 0 and t[-1] - 24*i > tf:
                X_1 = solve_ivp(model_light.Simulation, [t[0]+24*i,t[0]+24*i+tf], [X[-1],N[-1],P[-1],L[-1]], t_eval=t[24*i:24*i+tf])    
                X_2 = solve_ivp(model_dark.Simulation, [t[0]+24*i+tf,t[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t[24*i+tf:])
                X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                L = np.concatenate((L,X_1.y[2],X_2.y[2]))
                # X = X.append(X_2.y[0])
                # N = N.append(X_2.y[1])
            elif i!=0 and t[-1] - 24*j < 0 and t[-1] - 24*i <= tf:
                X_ = solve_ivp(model_light.Simulation, [t[0]+24*i,t[-1]], [X[-1],N[-1],P[-1],L[-1]], t_eval=t[24*i:])
                X = np.append(X,X_.y[0])
                N = np.append(N,X_.y[1])
                P = np.append(P,X_.y[2])
                L = np.append(L,X_.y[3])
    else:   
        cycle = np.ceil(t_exp[-1]/24)    # 光暗循环代数
        for m in model:
            if m.ligh == True:
                model_light = m
            if m.ligh == False:
                model_dark = m
                
        
        def cycle0():
            """
            第一个且仅有一个光暗循环，且不考虑脂质生产
            """
            global X
            global N
            global P
            if cycle == 0 and t_exp[-1] <= tf and model_light.lipid == False: 
                X_ = solve_ivp(model_light.Simulation, [t_exp[0],t_exp[-1]], [X0,N0,P0], t_eval=t_exp)
                X = X_.y[0]
                N = X_.y[1]
                P = X_.y[2]
                return [X,N,P]
            elif cycle == 0 and t_exp[-1] > tf and model_light.lipid == False:
                # 找到离tf最近的时间点
                t_1 = Find(tf, t_exp)
                X_1 = solve_ivp(model_light.Simulation, [t_exp[0],tf], [X0,N0,P0], t_eval=t_1[0][0])
                X_2 = solve_ivp(model_dark.Simulation,[tf,t_exp[-1]],[X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]],t_eval=t_1[1])
                X = np.append(X_1.y[0],X_2.y[0])
                N = np.append(X_1.y[1],X_2.y[1])
                P = np.append(X_1.y[2],X_2.y[2])
                 
                #return [X,N,P]
        def cycle_24():
            """
            第一个且仅有一个光暗循环，且光暗比正好是24：0，且不考虑脂质生产
            """
            if tf == 24 and model_light.lipid == False:
                X_ = solve_ivp(model_light.Simulation, [0,t_exp[-1]], [X0,N0,P0], t_eval=t_exp)
                X = X_.y[0]
                N = X_.y[1]
                P = X_.y[2]
                return [X,N,P]
            
        def cycle1():
            """
            非仅有一个光暗循环，且不考虑脂质生产
            """
            # if cycle !=0 and 23<=tf<24  and model_light.lipid == False:
            #     for i in range(int(cycle)):
            #         j = i+1
            #         if i==0: #and t[-1] - 24*j >= 0:
            #             # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
            #             t_01 = Find(24*j,t_exp)  # 找到当前循环所在区间的终点
            #             t_1 = Find(tf+24*i, t_exp[0:t_01[0][1]])
            #             print(t_1)
            #             X_1 = solve_ivp(model_light.Simulation, [t_exp[0],t_1[0][0][-1]], y0=[X0,N0,P0], t_eval=t_1[0][0])
            #             # model_dark.X0=X_1.y[0][-1]
            #             # model_dark.N0=X_1.y[1][-1]
            #             # model_dark.P0=X_1.y[2][-1]
            #             X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_01[1][-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
            #             X = np.append(X_1.y[0],X_2.y[0])
            #             N = np.append(X_1.y[1],X_2.y[1])
            #             P = np.append(X_1.y[2],X_2.y[2])
                        
            #         elif i!=0 and t_exp[-1] - 24*j > 0:
            #             # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
            #             t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
            #             t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
            #             t_1 = Find(tf+24*i, t_exp[t_0:t_01]) 
            #             X_1 = solve_ivp(model_light.Simulation, [t_exp[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
            #             # model_dark.X0=X_1.y[0][-1]
            #             # model_dark.N0=X_1.y[1][-1]
            #             # model_dark.P0=X_1.y[2][-1]
            #             X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[t_01]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
            #             X = np.concatenate((X,X_1.y[0],X_2.y[0]))
            #             #print(X.shape)
            #             N = np.concatenate((N,X_1.y[1],X_2.y[1]))
            #             P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                    
            #         elif i!=0 and t_exp[-1] - 24*j == 0:
            #             t_0 = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
            #             t_1 = Find(tf+24*i, t_exp[t_0:])
            #             X_1 = solve_ivp(model_light.Simulation, [t_exp[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
            #             # model_dark.X0=X_1.y[0][-1]
            #             # model_dark.N0=X_1.y[1][-1]
            #             # model_dark.P0=X_1.y[2][-1]
            #             X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
            #             X = np.concatenate((X,X_1.y[0],X_2.y[0]))
            #             #print(X.shape)
            #             if X.shape[0] != 121:
            #                 #print("补充")
            #                 X = np.append(X,np.zeros(121-X.shape[0]))
            #             N = np.concatenate((N,X_1.y[1],X_2.y[1]))
            #             P = np.concatenate((P,X_1.y[2],X_2.y[2]))
            #             # model_light.X0=X_1.y[0][-1]
            #             # model_light.N0=X_1.y[1][-1]
            #             # model_light.P0=X_1.y[2][-1]
                    
            #         elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i > tf:
            #             t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
            #             # t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
            #             t_1 = Find(tf+24*i, t_exp[t_0:])
            #             X_1 = solve_ivp(model_light.Simulation, [t[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
            #             # model_dark.X0=X_1.y[0][-1]
            #             # model_dark.N0=X_1.y[1][-1]
            #             # model_dark.P0=X_1.y[2][-1]
            #             X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
            #             X = np.concatenate((X,X_1.y[0],X_2.y[0]))
            #             N = np.concatenate((N,X_1.y[1],X_2.y[1]))
            #             P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                        
            #         elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i <= tf:
            #             t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
            #             X_ = solve_ivp(model_light.Simulation, [t_0,t_exp[-1]], [X[-1],N[-1],P[-1]], t_eval=t_exp[t_0:])
            #             X = np.append(X,X_.y[0])
            #             N = np.append(N,X_.y[1])
            #             P = np.append(P,X_.y[2])
            #     return [X,N,P]
            if cycle != 0 and tf != 24 and model_light.lipid == False:
                for i in range(int(cycle)):
                    j = i+1
                    if i==0: #and t[-1] - 24*j >= 0:
                        # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
                        t_01 = Find(24*j,t_exp)  # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[0:t_01[0][1]])
                        # print(t_1)
                        X_1 = solve_ivp(model_light.Simulation, [t_exp[0],t_1[0][0][-1]], y0=[X0,N0,P0], t_eval=t_1[0][0])
                        # model_dark.X0=X_1.y[0][-1]
                        # model_dark.N0=X_1.y[1][-1]
                        # model_dark.P0=X_1.y[2][-1]
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_01[1][-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
                        X = np.append(X_1.y[0],X_2.y[0])
                        N = np.append(X_1.y[1],X_2.y[1])
                        P = np.append(X_1.y[2],X_2.y[2])
                        
                    elif i!=0 and t_exp[-1] - 24*j > 0:
                        # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[t_0:t_01]) 
                        X_1 = solve_ivp(model_light.Simulation, [t_exp[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
                        # model_dark.X0=X_1.y[0][-1]
                        # model_dark.N0=X_1.y[1][-1]
                        # model_dark.P0=X_1.y[2][-1]
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[t_01]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
                        X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                        #print(X.shape)
                        N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                        P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                    
                    elif i!=0 and t_exp[-1] - 24*j == 0:
                        t_0 = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        t_1 = Find(tf+24*i, t_exp[t_0:])
                        X_1 = solve_ivp(model_light.Simulation, [t_exp[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
                        # model_dark.X0=X_1.y[0][-1]
                        # model_dark.N0=X_1.y[1][-1]
                        # model_dark.P0=X_1.y[2][-1]
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
                        X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                        #print(X.shape)
                        if X.shape[0] != 121:
                            #print("补充")
                            X = np.append(X,np.zeros(num-X.shape[0]))
                        N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                        P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                        # model_light.X0=X_1.y[0][-1]
                        # model_light.N0=X_1.y[1][-1]
                        # model_light.P0=X_1.y[2][-1]
                    
                    elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i > tf:
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        # t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[t_0:])
                        X_1 = solve_ivp(model_light.Simulation, [t[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1]], t_eval=t_1[0][0])    
                        # model_dark.X0=X_1.y[0][-1]
                        # model_dark.N0=X_1.y[1][-1]
                        # model_dark.P0=X_1.y[2][-1]
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1]], t_eval=t_1[1])
                        X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                        N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                        P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                        
                    elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i <= tf:
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        X_ = solve_ivp(model_light.Simulation, [t_0,t_exp[-1]], [X[-1],N[-1],P[-1]], t_eval=t_exp[t_0:])
                        X = np.append(X,X_.y[0])
                        N = np.append(N,X_.y[1])
                        P = np.append(P,X_.y[2])
                return [X,N,P]
                    
        def cycle0_lipid():
            """
            第一个且仅有一个光暗循环，且考虑脂质生产
            """
            global X
            global N
            global P
            global L
        
            if cycle == 0 and t_exp[-1] <= tf and model_light.lipid == True:
                X_ = solve_ivp(model_light.Simulation, [t_exp[0],t_exp[-1]], [X0,N0,P0,L0], t_eval=t_exp)
                X = X_.y[0]
                N = X_.y[1]
                P = X_.y[2]
                L = X_.y[3]
                return [X,N,P,L]
            elif cycle == 0 and t_exp[-1] > tf and model_light.lipid == True:
                # 找到离tf最近的时间点
                t_1 = Find(tf, t_exp)
                X_1 = solve_ivp(model_light.Simulation, [t_exp[0],tf], [X0,N0,P0,L0], t_eval=t_1[0][0])
                X_2 = solve_ivp(model_dark.Simulation,[tf,t_exp[-1]],[X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]],t_eval=t_1[1])
                X = np.append(X_1.y[0],X_2.y[0])
                N = np.append(X_1.y[1],X_2.y[1])
                P = np.append(X_1.y[2],X_2.y[2])
                L = np.append(X_1.y[3],X_2.y[3])
                return [X,N,P,L]
        def cycle_24_lipid():
            """
            第一个光暗循环，且仿真时间正好是24小时，且考虑脂质生产
            """
            if tf == 24 and model_light.lipid == True:
                X_ = solve_ivp(model_light.Simulation, [0,t_exp[-1]], [X0,N0,P0,L0], t_eval=t_exp)
                X = X_.y[0]
                N = X_.y[1]
                P = X_.y[2]
                L = X_.y[3]
               # return [X,N,P,L]
        def cycle1_lipid():
            """
            非仅有一个光暗循环中第一个循环，且考虑脂质生产
            """
            if cycle != 0 and tf!=24 and model_light.lipid == True:
                for i in range(int(cycle)):
                    j = i+1
                    if i==0: #and t[-1] - 24*j >= 0:
                        # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
                        t_01 = Find(24*j,t_exp)  # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[0:t_01[0][1]])
                        X_1 = solve_ivp(model_light.Simulation, [t_exp[0],t_1[0][0][-1]], y0=[X0,N0,P0,L0], t_eval=t_1[0][0])
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_01[1][-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t_1[1])
                        X = np.append(X_1.y[0],X_2.y[0])
                        N = np.append(X_1.y[1],X_2.y[1])
                        P = np.append(X_1.y[2],X_2.y[2])
                        L = np.append(X_1.y[3],X_2.y[3])
                    elif i!=0 and t_exp[-1] - 24*j >= 0:
                        # t_1 = heapq.nsmallest(tf+24*i, t_exp[24*i,24*j])
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[t_0:t_01]) 
                        X_1 = solve_ivp(model_light.Simulation, [t_exp[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1],L[-1]], t_eval=t_1[0][0])    
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[t_01]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t_1[1])
                        X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                        N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                        P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                        L = np.concatenate((L,X_1.y[3],X_2.y[3]))
                        
                    elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i > tf:
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        # t_01 = Find(24*j,t_exp)[0][1] # 找到当前循环所在区间的终点
                        t_1 = Find(tf+24*i, t_exp[t_0:])
                        X_1 = solve_ivp(model_light.Simulation, [t[t_0],t_1[0][0][-1]], [X[-1],N[-1],P[-1],L[-1]], t_eval=t_1[0][0])    
                        X_2 = solve_ivp(model_dark.Simulation, [t_1[0][0][-1],t_exp[-1]], [X_1.y[0][-1],X_1.y[1][-1],X_1.y[2][-1],X_1.y[3][-1]], t_eval=t_1[1])
                        X = np.concatenate((X,X_1.y[0],X_2.y[0]))
                        N = np.concatenate((N,X_1.y[1],X_2.y[1]))
                        P = np.concatenate((P,X_1.y[2],X_2.y[2]))
                        L = np.concatenate((L,X_1.y[3],X_2.y[3]))
                    elif i!=0 and t_exp[-1] - 24*j < 0 and t_exp[-1] - 24*i <= tf:
                        t_0  = Find(24*i,t_exp)[0][1] # 上一轮循环的终点，作为这一轮循环的起点
                        X_ = solve_ivp(model_light.Simulation, [t_0,t_exp[-1]], [X[-1],N[-1],P[-1],L[-1]], t_eval=t_exp[t_0:])
                        X = np.append(X,X_.y[0])
                        N = np.append(N,X_.y[1])
                        P = np.append(P,X_.y[2])
                        L = np.append(L,X_.y[3])
                        
                #return [X,N,P,L] 
                    
        properties = [cycle0,cycle_24,cycle1,cycle0_lipid,cycle_24_lipid,cycle1_lipid] 
        X__ = []
        for func in properties:
            X__.append(func())
        #return a for a in 
        for i in X__:
            if type(i) == list:
                return i
    
