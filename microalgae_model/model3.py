import numpy as np
import random
import sympy as sp
# from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 声明变量生物量浓度'X'，时间't'，氮元素消耗量'N'
# t,N = sp.symbols('x,t,N')
# X = sp.symbols('X', cls=sp.Function)
# parameter = {"xmax":0.5, # 环境最大浓度
#              "um":0.75,  # 最大比增长率
#              "Ka":0.0114, # 光衰减系数
#              "ki":20,
#              "ks":2.0
#             }
#%% 模型建立
def Inititial(parameters):
    "若给了初始参数，则将字典形式初始列表转化为列表"
    if isinstance(parameters,dict):
        parameter = []
        for par in parameters:
            parameter.append(parameters[par])
        return parameter
    if isinstance(parameters,list):
        parameter = parameters
        return parameter
    if isinstance(parameters,np.ndarray):
        if parameters.ndim != 1:
            parameter = parameters.reshape(-1,)
        else:
            parameter = parameters
        return parameter



class BioModels(object):
    def __init__(self,X0,I0,N0,tf,P0,name = None,lipid=False):
        
        self.X0 = X0 # 初始生物量浓度
        self.I0 = I0 # 入射光照强度
        self.N0 = N0 # 初始N元素浓度
        self.tf = tf # 培养时间
        # self.CO_2 = CO_2 # co2浓度
        self.P0 = P0 # 初始P元素浓度
        self.Name = name
        self.parameter = None
        self.lipid = lipid
    def Inititial(self,parameters):
        "若给了初始参数，则将字典形式初始列表转化为列表"
        if isinstance(parameters,dict):
            self.parameter = []
            for par in parameters:
                self.parameter.append(parameters[par])
            return self.parameter
        if isinstance(parameters,list):
            self.parameter = parameters
            return self.parameter
    
    
    def Specific_growth_rate(self,X,t,N):
        if self.name == None:
            raise ValueError(
                "the name of the class of model cannot be None")
            
        if isinstance(self.name,str):
            raise ValueError(
                "the type of the attitude:name, must be a string")
        if self.name == "Logistic":
            self.mu = X * (1-X/self.Xmax)
     
            
     
        
class Logistic(BioModels):
    def __init__(self, X0, I0, N0,  tf,P0,parameters=None,ligh = True,z=0.06,name = "Logistic",lipid=False):
        super(Logistic, self).__init__(X0, I0, N0, tf,  P0,name)
        # self.ligh = ligh
        # if self.ligh:
        #     self.parameter = Inititial(parameters_light)
        #     self.z = z
        # if not self.ligh:
        #     self.parameter = Inititial(parameters_dark)
        self.ligh = ligh
        # if self.ligh:
        #     self.parameter = Inititial(parameters_light)
        #     self.L = L
        # if not self.ligh:
        #     self.parameter = Inititial(parameters_dark)
        self.parameter = parameters
        self.z = z
    
        
        
    def Simulation(self,t,var):
        X = var[0] 
        N = var[1]
        P = var[2]
        # 十步求解法求解平均光强下的比增长率
        # self. = "um"
        if self.ligh:   
            
            # L = var[2]
            # Q = var[3]
            # Sr = self.X0 / self.N0
            # QL = var[2]
            # QC = var[3]
            # QP = var[4]
            def light():
                global I
                # mu_I_average = 0
                # for i in range(1,10):
                #    mu_I_average = mu_I_average + self.parameter["um"]/20 *((self.I0/(self.I0 + self.parameter["KIs"] + self.I0**2/self.parameter["KIi"])) +\
                #                                                         2*(self.I0 * np.exp(-self.parameter["Ka"]*X*i*self.z/10))/((self.I0 * np.exp(-self.parameter["Ka"]*X*i*self.z/10))+\
                #                                                         self.parameter["KIs"] + (self.I0 * np.exp(-self.parameter["Ka"]*X*10*self.z/10))**2/self.parameter["KIi"])+\
                #                                                         (self.I0 * np.exp(-self.parameter["Ka"]*X*self.z))/((self.I0 * np.exp(-self.parameter["Ka"]*X*self.z))+\
                #                                                         self.parameter["KIs"] + (self.I0 * np.exp(-self.parameter["Ka"]*X*self.z))**2/self.parameter["KIi"]))
                # mu_I_average = []
                # for xi in X:
                #     if xi >= 0:
                #         I = self.I0*(1-np.exp(-self.parameter["Ka"]*xi*self.z))/(self.parameter["Ka"]*xi*self.z)
                #         mu_I_average.append(self.parameter["um"] * I /(self.parameter["KIs"] + I + I**2/self.parameter["KIi"]))
                # else:
                #     I = self.I0*(1-np.exp(-self.parameter["Ka"]*X*self.z))/(self.parameter["Ka"]*X*self.z)
                I = self.I0*(1-np.exp(-self.parameter["Ka"]*X*self.z))/(self.parameter["Ka"]*X*self.z)
                mu_I_average = self.parameter["um"] * I /(self.parameter["KIs"] + I + I**2/self.parameter["KIi"])
                return mu_I_average
            # 在无外部碳源的条件下CO2对微藻生长的影响
            def CO2_l():
                pass
            # 磷元素作为有机碳对微藻生长的影响 https://doi.org/10.1186/s13068-021-01912-2 eq(4)
            def Phosphorus():
                return P / (P + self.parameter["KPs"] + P**2/self.parameter["KPi"])
            # N对微藻生长的影响
            def Nitrogen():
                return N / (N + self.parameter["KNs"] + N**2/self.parameter["KNi"])
            # 温度对微藻生长的影响
            def Temperature():
                pass
            # xmax的限制
            def xmaxlim():
                
                # xmax_ = self.parameter["xmax"]*X/(X+X**2/(X+self.parameter["xmax"]))
                xmax_ = self.parameter["xmax"] * N/(self.parameter["KNs"]/3+N) * P/(self.parameter["KPs"]/3+P) * \
                        I /(self.parameter["KIs"] + I + I**2/self.parameter["KIi"])
                return xmax_
            # 模拟生物量的变化
            dxdt = light() * Nitrogen() * Phosphorus() * (1-X/xmaxlim())*X - self.parameter["ud"]*X
            # 模拟外部氮元素含量的变化
            # dNdt = -light()*Nitrogen()*(1 - X/self.parameter[0]) *X * self.parameter[8]
            dNdt = -dxdt * 1/self.parameter["Yx/N"] - self.parameter["m/N"]*X
            # 模拟外部P元素含量的变化
            # dNdt = -light()*Nitrogen()*(1 - X/self.parameter[0]) *X * self.parameter[8]
            dPdt = -dxdt * 1/self.parameter["Yx/P"] - self.parameter["m/P"]*X
            # # 模拟细胞内部氮元素配额Q的变化  10.1002/bit.26744
            # dQdt = -dNdt/self.parameter[8] - light()*Nitrogen()*(Q-self.parameter[13])
            # # 模拟脂质生产(非氮胁迫条件下) https://doi.org/10.1007/s10811-016-0841-4
            # if dQdt>=0 :  # 如果处于非胁迫状态
            #     # dLdt = self.parameter[9]*(light()*Nitrogen()*(1 - X/self.parameter[0]) *X ) - self.parameter[10] * X
            #     dLdt = self.parameter[9]*dxdt - self.parameter[10] * X
            # if dQdt<0:
            #     dLdt = self.parameter[11]*dxdt - self.parameter[12] * Sr * (-dQdt)
            # # 细胞水平模拟脂质生产  10.1002/bit.26744
            # dQLdt = self.parameter[11]*self.parameter[12]*QC + self.parameter[13]*
            if self.lipid:
                # 模拟脂质含量的变换 10.1007/s10811-016-0841-4  Kinetic modeling of mixotrophic growth of Chlorella vulgaris as a new feedstock for biolubricant
                dLdt = -self.parameter["alpha"]* dxdt + self.parameter["beta"]*X
                return [dxdt,dNdt,dPdt,dLdt]
            else:
                return [dxdt,dNdt,dPdt]
        if not self.ligh:
            # X = var[0]
            # N = var[1]
            # L = var[2]
            # Q = var[3]
            # Sr = self.X0 / self.N0
            # 乙酸盐作为有机碳对微藻生长的影响 https://doi.org/10.1186/s13068-021-01912-2 eq(4)
            def Phosphorus():
                return P / (P + self.parameter["KPs"] + P**2/self.parameter["KPi"])
            
            def Nitrogen():
                return N / (N + self.parameter["KNs"] + N**2/self.parameter["KNi"])
            # xmax的限制
            def xmaxlim():
                
                xmax_ = self.parameter["xmax"]*X/(X+X**2/(X+self.parameter["xmax"]))
                return xmax_
            
            dxdt = Nitrogen() * Phosphorus() * (1-X/xmaxlim())*X - self.parameter["ud"]*X
            dNdt = dxdt * 1/self.parameter["Yx/N"] - self.parameter["m/N"]*X
            dPdt = dxdt * 1/self.parameter["Yx/P"] - self.parameter["m/P"]*X
            #dCdt = -dxdt *  1/self.parameter["Yx/A"]
            
            if self.lipid:
                # 模拟脂质含量的变换 10.1007/s10811-016-0841-4  Kinetic modeling of mixotrophic growth of Chlorella vulgaris as a new feedstock for biolubricant
                dLdt = -self.parameter["alpha"]* dxdt + self.parameter["beta"]*X
                return [dxdt,dNdt,dPdt,dLdt]
            else:
                return [dxdt,dNdt,dPdt]
    
    
    
# class Heterotrophism(BioModels):
#     def __init__(self, X0, N0, tf, CO_2, T0,parameters=None,name = "Heterotrophism"):
#         super(Heterotrophism,self).__init__(X0, N0, tf, CO_2, T0,name)
#         self.parameter = Inititial(parameters)
    
#     # parameter = {"unmax":3.5,  # 氮吸收最大速率  0
#     #              "KNi":600,    # 营养抑制常数  1
#     #              "KNs":300.0,   # 营养饱和常数  2
#     #              "Yx/N":200   # N元素吸收速率 3
#     #              "KIs":100.0 ,  # 光饱和常数    4
#     #              "ud":0.015,   # 比光合自养衰减率 5
#     #              "KNi":600,    # 营养抑制常数  6
#     #              "KNs":300.0,   # 营养饱和常数  7
#     #              "Yx/N":200   # N元素吸收速率 8
#     #             }    
    
    
#     def Simulation(self,t,var):
        
#         X = var[0]
#         N = var[1]
#         L = var[2]
#         Q = var[3]
#         Sr = self.X0 / self.N0
#         def Nitrogen():
#             return N / (N + self.parameter[2] + N**2/self.parameter[1])
        
#         def Phosphorus():
#             pass
#         dNdt = -self.parameter[0] * Nitrogen() * self.parameter[3] * X
#         dxdt = self.parameter[0] * Nitrogen() * X
#         # 模拟细胞内部氮元素配合Q的变化
#         dQdt = -dNdt/self.parameter[8] - Nitrogen()*(Q-self.parameter[13])
#         # 模拟脂质生产(非氮胁迫条件下) https://doi.org/10.1007/s10811-016-0841-4
#         if dQdt>=0 :  # 如果处于非胁迫状态
#             # dLdt = self.parameter[9]*(light()*Nitrogen()*(1 - X/self.parameter[0]) *X ) - self.parameter[10] * X
#             dLdt = self.parameter[9]*dxdt - self.parameter[10] * X
#         if dQdt<0:
#             dLdt = self.parameter[11]*dxdt - self.parameter[12] * Sr * (-dQdt)
#         # dLdt = self.parameter[4]*(self.parameter[0] * Nitrogen() * X) + self.parameter[5] * X
#         return [dxdt,dNdt,dLdt]
        

    

#%% 参数优化
# import geatpy as ea
# from myproblem3 import MyProblem3
# def param_estimation(model,NIND,I0,X0,S0,tf,z,X_exp,t):
#     problem = MyProblem3(model,NIND,I0,X0,S0,tf,z,X_exp,t)
    
#     algorithm = ea.soea_EGA_templet(problem,
#                                 ea.Population(Encoding='RI',NIND=NIND),
#                                 MAXGEN=100,  # 最大进化代数
#                                 logTras=1
#                                 )  
#     # algorithm.mutOper.F = 0.5  # 差分进化中的参数F
#     # algorithm.recOper.XOVR = 0.7  # 重组概率
#     res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=False, drawLog=False, saveFlag=False)
    
#     return res
    
#     pass

#%% 主函数
# if __name__ == '__main__':
    
#     tf = 12 #14:10光暗比
#     X0=0.03895
#     N0=200
#     L0 = 0.0142 # 随便写的初始脂质产量
#     I0=425
#     z = 0.06
#     t0 = [X0,N0]
    
#     parameter_light = {"xmax":3.5,  # 环境最大浓度  0
#                  "um":0.95,   # 最大比增长率  1
#                  "Ka":0.0114, # 光衰减系数    2
#                  "KIi":250,    # 光抑制常数    3
#                  "KIs":100.0 ,  # 光饱和常数    4
#                  "ud":0.015,   # 比光合自养衰减率 5
#                  "KNi":600,    # 营养抑制常数  6
#                  "KNs":300.0,   # 营养饱和常数  7
#                  "Yx/N":200,   # N元素吸收速率 8
#                  "alpha":0.46,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）9
#                   "beta":0.0005, # 产品(脂质)的具体形成速率 10 https://doi.org/10.1007/s10811-016-0841-4 10
#                   # 中央通路模型10.1002/bit.26744
#                  # "Yc2L":1.000,  #   gL/gC   11
#                  # "uc":0.01,     #   1/h  12
#                  # "YP2L": 1.000, #   gL/gN   13
#                  # "DP":0.001     #   1/h 14
#                  # "uL": 1.0    # gL/10^10 cell 15
#                  # 细胞配额模型http://dx.doi.org/10.1016/j.biortech.2016.07.063
#                  "alpha2":0.46,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）11
#                   "beta2":0.0005, # 产品(脂质)的具体形成速率 12 
#                   "q0":  9.79   # mg N/cell*10^9(或者L)  13
#                 }
#     parameter_dark= {"unmax":0.75,  # 氮吸收最大速率  0
#                  "KNi":800,    # 营养抑制常数  1
#                   "KNs":300.0,   # 营养饱和常数  2
#                  "Yx/N":100 ,  # N元素吸收速率 3
#                  "alpha":0.46,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）4
#                  "beta":0.0005, # 产品(脂质)的具体形成速率 10 https://doi.org/10.1007/s10811-016-0841-4 5
#                  # 细胞配额模型http://dx.doi.org/10.1016/j.biortech.2016.07.063
#                  "alpha2":0.46,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）6
#                   "beta2":0.0005, # 产品(脂质)的具体形成速率 7 
#                   "q0":  9.79   # mg N/cell*10^9(或者L)  8
#                      }
#     t = np.linspace(0,100,num=101)
#     t_ = t[0]
    
#     model_light = Logistic(X0=0.01895, I0=25, N0=0.02,tf=tf, T0=28,parameters_light=parameter_light,ligh = True,L=0.06)
#     # model_dark = Heterotrophism(X0=0.01895, N0=0.02, tf=tf, T0=28,parameters=parameter_dark)
#     model_dark = Logistic(X0=0.01895, I0 = 25, N0=0.02, tf=tf, T0=28,parameters_dark=parameter_dark,ligh = False)
#     # X  = solve_ivp(model_light.Simulation, [0,tf], t0,t_eval=t)
    
    
#     cycle = np.ceil(t[-1]/24)    # 光暗循环代数
    
#     for i in range(int(cycle)):
#         j = i+1
#         if cycle == 0:
#             if t[-1] <= tf:
#                 X_ = solve_ivp(model_light.Simulation, [0,t[-1]], t0, t_eval=t)
#                 X = X_.y[0]
#                 N = X_.y[1]
#                 L = X_.y[2]
#             else:
#                 X_1 = solve_ivp(model_light.Simulation, [0,tf], t0, t_eval=t[:tf])
#                 X_2 = solve_ivp(model_dark.Simulation,[tf,t[-1]],[X_1.y[0][-1],X_1.y[1][-1]],t_eval=t[tf:])
#                 X = np.append(X_1.y[0],X_2.y[0])
#                 N = np.append(X_1.y[1],X_2.y[1])
#                 # L = np.append(X_1.y[2],X_2.y[2])
#                 # X = X.append(X_2.y[0])
#                 # N = N.append(X_2.y[1])
#         elif i==0: #and t[-1] - 24*j >= 0:
#             X_1 = solve_ivp(model_light.Simulation, [t[0],t[0]+tf], t0, t_eval=t[:tf])
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
    
    
#     # 生成实验数据
#     X_exp = X + random.gauss(0,0.016)
    
#     # res = param_estimation(model,NIND=20,I0=I0,X0=X0,S0=N0,tf = tf,z = z,X_exp=X_exp,t=t)
    
#     # 计算比增长率
#     miu = []
#     for i in range(1,len(X)):
#         miu.append(np.log(X[i]/X[i-1])/(t[i]-t[i-1]))  
#     # 计算生物量产率
#     dxdt = []
#     for i in range(1,len(X)):
#         dxdt.append((X[i]-X[i-1])/(t[i]-t[i-1]))  
#     # 计算N的消耗
#     # N  = -parameter["Yx/N"] * (X -X0 )+ N0
#     fig, ax = plt.subplots(figsize = (12,8))
#     # ax1=fig.add_subplot(111, label="1")
#     # ax1.set_xlabel('Time',fontsize=30)
#     # ax1.set_ylabel('X',rotation = 0, labelpad = 40,fontsize=30)
#     # ax1.plot(t,X, color='b', label='X')
#     # ax1.scatter(t,X_exp, color='b', label='X')
#     # ax1.legend(loc='best',prop={"size":20})
#     plt.plot(t,X, color='b', label='X')
#     plt.scatter(t,X_exp, color='b', label='X')
    
#     # fig, ax = plt.subplots(figsize = (12,8))
#     # plt.plot(t,L, color='orange', label='L')
#     # plt.scatter(t,X_exp, color='b', label='X')
#     # ax2 = ax1.twinx()
#     # ax2.plot(t[1:],dxdt, color='orange', label = 'dxdt')
#     # ax2.legend(loc='best',prop={"size":20})
#     # ax2.set_xlabel('Time',fontsize=30)
#     # ax2.set_ylabel('dxdt',rotation = 0, labelpad = 40,fontsize=30)
#     fig, ax = plt.subplots(figsize = (12,8))
#     plt.plot(t,N, color='g', label='N')
#     # plt.plot(t[1:],miu, color='r', label = 'miu')
#     plt.legend(loc='best',prop={"size":20});
#     plt.xlabel('Time',fontsize=30)
#     plt.ylabel('N',rotation = 0, labelpad = 40,fontsize=30)
    
    
#     # # 优化后的参数建模
#     # res_light = param_estimation(model,NIND=20,I0=I0,X0=X0,S0=N0,tf = tf,z = z,X_exp=X_exp,t=t)
#     # param_res = res["Vars"].reshape(-1,)
#     # model2 = Logistic(X0=0.01895, I0=25, N0=0.02, tf=tf, CO_2=0.02, T0=28,parameters=param_res,L=0.06)
#     # X_2  = solve_ivp(model2.Simulation, [0,tf], t0,t_eval=t)
#     # # 计算比增长率
#     # miu_2 = []
#     # for i in range(1,len(X_2)):
#     #     miu_2.append(np.log(X_2.y[0][i]/X_2.y[0][i-1])/(t[i]-t[i-1]))  
#     # # 计算生物量产率
#     # dxdt_2 = []
#     # for i in range(1,len(X_2.y[0])):
#     #     dxdt_2.append((X_2.y[0][i]-X_2.y[0][i-1])/(t[i]-t[i-1]))  
#     # # 计算N的消耗
#     # # N_2  = -param_res[-1] * (X_2 -X0 )+ N0    
#     # N_2 = X_2.y[1]
#     # fig, ax = plt.subplots(figsize = (12,8))
#     # ax3=fig.add_subplot(111, label="1")
#     # ax3.set_ylabel('X_2',rotation = 0, labelpad = 40,fontsize=30)
#     # ax3.plot(t,X_2.y[0], color='b', label='X_2')
#     # ax3.scatter(t,X_exp, color='b', label='X_exp')
#     # ax3.legend(loc='best',prop={"size":20})
#     # ax4 = ax1.twinx()
#     # ax4.plot(t[1:],dxdt_2, color='orange', label = 'dxdt_2')
#     # ax4.legend(loc='best',prop={"size":20})
#     # ax4.set_xlabel('Time',fontsize=30)
#     # ax4.set_ylabel('dxdt_2',rotation = 0, labelpad = 40,fontsize=30)
#     # fig, ax = plt.subplots(figsize = (12,8))
#     # plt.plot(t,N_2, color='g', label='N_2')
#     # # plt.plot(t[1:],miu, color='r', label = 'miu')
#     # plt.legend(loc='best',prop={"size":20});
#     # plt.xlabel('Time',fontsize=30)
#     # plt.ylabel('N_2',rotation = 0, labelpad = 40,fontsize=30)
    


