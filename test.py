import numpy as np
from microalgae_model.model3 import Logistic
from microalgae_model.cycle import Cycle
import pandas as pd
import matplotlib.pyplot as plt
# from integrated_mc import convolute_RF,convolute_K,convolute_RF_algae
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal,DiscreteLaplace,Poisson
from golem import * 
from skopt import Space
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from skopt.learning import RandomForestRegressor as RF_std
# algae_curve_data_2 = []
# for i,data in enumerate(algae_curve_data):
#     if (i%11 == 0) and (i != len(algae_curve_data)-1):
#         algae_curve_data_2.append(data[:-1])
#         for j in range(10):
#             algae_curve_data_2[-1] = np.append(algae_curve_data_2[-1],algae_curve_data[i+j,-1])
#             # algae_curve_data_2[-1] = np.insert(algae_curve_data_2[-1],-1,algae_curve_data[i+j,-1])
# df  = pd.DataFrame(algae_curve_data_2)
# df.to_excel("algae_curve_data_3.xlsx")
#%% 生成推荐条件的微藻生长曲线
# ini_param = [0.1138,121,0.06]
# t = np.linspace(0, 120,121)
# def microalgae_growth(ini_param,var,t_up=120,t_lb=0):
#     # ini_param=[X0,num,z]
#     parameter_light = {"xmax":0.5193,  # 环境最大浓度  ,g/m3
#                  "um":45.9946,   # 最大比增长率     ,1/h
#                  "Ka":0.054, # 光衰减系数        ,m2/g
#                  "KIi":987.882,    # 光抑制常数    3            ,μmol/m2 s
#                  "KIs":21.7205 ,  # 光饱和常数    4           ,μmol/m2 s
#                  "ud":0.0001,   # 比光合自养衰减率 5
#                  "KNi":853.5633,    # N元素营养抑制常数  6       ,mg/L
#                  "KNs":90.502,   # N元素营养饱和常数  7      ,mg/L
#                  "Yx/N":48.871,   # N元素吸收速率 8            ,无量纲
#                    "m/N":7.6401,     # N元素相关系数
#                  #"alpha":750,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）9
#                   #"beta":0.0005, # 产品(脂质)的具体形成速率 https://doi.org/10.1007/s10811-016-0841-4 
#                    "KPi":991.1707 ,    # C元素营养抑制常数  9
#                   "KPs":890.32927,   # C元素营养饱和常数  10
#                    "Yx/P":87.678067,   # 乙酸吸收速率 18
#                    "m/P":1.042953
#                 }
#     parameter_dark= {"xmax":0.3958137,  # 环境最大浓度  ,g/m3
                     
#                     "um":40.7287343607939,  # 氮吸收最大速率  0
#                      "ud":0.0001,   # 比光合自养衰减率 5
#                  "KNi":173.31,    # 营养抑制常数  1
#                   "KNs":130.22,   # 营养饱和常数  2
#                  "Yx/N":70.6983 ,  # N元素吸收速率 3
#                   "m/N":34.18317,     # N元素相关系数
#                 # "alpha":91.235,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）4
#                  #"beta":39.596, # 产品(脂质)的具体形成速率 10 https://doi.org/10.1007/s10811-016-0841-4 5
#                   "KPi":38.65422 ,    # C元素营养抑制常数  9
#                   "KPs":38.2133,   # C元素营养饱和常数  10
#                   "Yx/P":59.0748,   # 乙酸吸收速率 13
#                    "m/P":0.0001
#                      }
#     t = np.linspace(0, 120,ini_param[1])
#     model_light = Logistic(X0=ini_param[0], I0=var[0], N0=var[1],tf=var[3], P0=var[2],parameters=parameter_light,ligh = True,z=ini_param[2])
#     model_dark = Logistic(X0=ini_param[0], I0=var[0], N0=var[1],tf=var[3], P0=var[2],parameters=parameter_dark,ligh = False)
#     model = [model_light,model_dark]
#     parameter = [parameter_light,parameter_dark]
#     [X,N,P] = Cycle(X0=ini_param[0],N0=var[1],P0=var[2],model=model,tf=var[3],t_up=t_up,t_lb=t_lb,num=ini_param[1],t_exp=t)
#     X1 = [X[i] for i in range(121) if i%12==0 ]
#     N1 = [N[i] for i in range(121) if i%12==0 ]
#     P1 = [P[i] for i in range(121) if i%12==0 ]
#     return [X[-4],N[-4],P[-4]],[X1,N1,P1],[X,N,P]

# X_samples_RF = pd.read_excel("X_samples_RF_df.xlsx").to_numpy()
# X_samples_RF = X_samples_RF[:,1:] # 去掉索引
# Y_samples_RF = pd.read_excel("y_samples_RF_df.xlsx").to_numpy()
# Y_samples_RF = Y_samples_RF[:,1:] # 去掉索引

# X     = []
# X_temp= 0
# X_max = 0
# i_max = 0
# for i,x in enumerate(X_samples_RF):    
#     X_ = microalgae_growth(ini_param,var=x,t_up=120,t_lb=0)
#     X.append(X_[2][0])
#     if X_[0][0] > X_temp:
#         X_temp = X_[0][0]
#         X_max = x
#         i_max = i
# print(X_max,i_max)  
# # 生长曲线  
# plt.figure()
# plt.plot(t,X[i_max])
# plt.show()


#%% 推荐条件生长曲线的预测不确定性
# space2=Space([(26,78),(0.335,0.385),(0.0064,0.025),(12,24)])
# D = 4
# bounds = np.array(space2.bounds).T
# bounds = bounds.tolist()
# dists = [DiscreteLaplace(2),Normal(0.012),Normal(0.001),DiscreteLaplace(0.1)] # 第一个维度上是t，但是t只参与预测，不参与优化
# model = Golem(goal='max', ntrees=4,random_state=42, nproc=1)
# model.fit(X=X_samples_RF, y=Y_samples_RF)
# y_robust = golem.predict(X=X_max.reshape(-1,1), distributions=dists)


#%% 不考虑输入参数不确定性的
algae_data_curve_2 = pd.read_excel('algae_data_curve_2.xlsx')
algae_data_curve_2 = np.array(algae_data_curve_2)
train_x_curve = algae_data_curve_2[:,2:6] # 目标域训练输入
train_y_curve = algae_data_curve_2[:,6]  # 目标与训练标签 微藻生物量
ini_param = [0.1138,121,0.06]
def microalgae_growth(ini_param,var,t_up=120,t_lb=0):
    # ini_param=[X0,num,z]
    parameter_light = {"xmax":0.5193,  # 环境最大浓度  ,g/m3
                 "um":45.9946,   # 最大比增长率     ,1/h
                 "Ka":0.054, # 光衰减系数        ,m2/g
                 "KIi":987.882,    # 光抑制常数    3            ,μmol/m2 s
                 "KIs":21.7205 ,  # 光饱和常数    4           ,μmol/m2 s
                 "ud":0.0001,   # 比光合自养衰减率 5
                 "KNi":853.5633,    # N元素营养抑制常数  6       ,mg/L
                 "KNs":90.502,   # N元素营养饱和常数  7      ,mg/L
                 "Yx/N":48.871,   # N元素吸收速率 8            ,无量纲
                   "m/N":7.6401,     # N元素相关系数
                 #"alpha":750,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）9
                  #"beta":0.0005, # 产品(脂质)的具体形成速率 https://doi.org/10.1007/s10811-016-0841-4 
                   "KPi":991.1707 ,    # C元素营养抑制常数  9
                  "KPs":890.32927,   # C元素营养饱和常数  10
                   "Yx/P":87.678067,   # 乙酸吸收速率 18
                   "m/P":1.042953
                }
    parameter_dark= {"xmax":0.3958137,  # 环境最大浓度  ,g/m3
                     
                    "um":40.7287343607939,  # 氮吸收最大速率  0
                     "ud":0.0001,   # 比光合自养衰减率 5
                 "KNi":173.31,    # 营养抑制常数  1
                  "KNs":130.22,   # 营养饱和常数  2
                 "Yx/N":70.6983 ,  # N元素吸收速率 3
                  "m/N":34.18317,     # N元素相关系数
                # "alpha":91.235,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）4
                 #"beta":39.596, # 产品(脂质)的具体形成速率 10 https://doi.org/10.1007/s10811-016-0841-4 5
                  "KPi":38.65422 ,    # C元素营养抑制常数  9
                  "KPs":38.2133,   # C元素营养饱和常数  10
                  "Yx/P":59.0748,   # 乙酸吸收速率 13
                   "m/P":0.0001
                     }
    t = np.linspace(0, 120,ini_param[1])
    model_light = Logistic(X0=ini_param[0], I0=var[0], N0=var[1],tf=var[3], P0=var[2],parameters=parameter_light,ligh = True,z=ini_param[2])
    model_dark = Logistic(X0=ini_param[0], I0=var[0], N0=var[1],tf=var[3], P0=var[2],parameters=parameter_dark,ligh = False)
    model = [model_light,model_dark]
    parameter = [parameter_light,parameter_dark]
    [X,N,P] = Cycle(X0=ini_param[0],N0=var[1],P0=var[2],model=model,tf=var[3],t_up=t_up,t_lb=t_lb,num=ini_param[1],t_exp=t)
    X1 = [X[i] for i in range(121) if i%12==0 ]
    N1 = [N[i] for i in range(121) if i%12==0 ]
    P1 = [P[i] for i in range(121) if i%12==0 ]
    return [X[-1],N[-1],P[-1]],[X1,N1,P1]

import geatpy as ea
from myproblem_nouncert import MyProblem

def EI_optimize(model,model_RF,bounds,distributions,model_type,y_):
    
    # model,model_RF,bounds,distribution,model_type = args
    
    problem = MyProblem(model,model_RF,bounds,distributions,model_type,y_)
    
    algorithm = ea.soea_EGA_templet(problem,
                                ea.Population(Encoding='RI',NIND=15),
                                MAXGEN=100,  # 最大进化代数
                                logTras=1
                                )  
    # algorithm.mutOper.F = 0.5  # 差分进化中的参数F
    # algorithm.recOper.XOVR = 0.7  # 重组概率
    res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=False, drawLog=False, saveFlag=False,)
    
    return res

# 定义采集函数取最大的函数

def propose_location(model,model_RF, X_sample, Y_sample, bounds,distributions=None,model_type='gp', n_restarts = 2):
    
    dim = X_sample.shape[1]   # X_sample: Sample locations (n x d). 所以dim = 1
    min_val = 1
    min_x = None
    mu,std = model_RF.predict(X_sample,return_std=True)
    y = mu
    y_ = np.max(y)
    EI_res = EI_optimize(model,model_RF,bounds,distributions,model_type,y_)  
    min_x = EI_res["Vars"]        
            
    return min_x.reshape(1, -1)
def robust_optimization_GW_3(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,X_samples=None,y_samples = None,model_type='gp',nproc=1):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
        model_RF = RF_std(n_estimators=4,criterion='squared_error')
    if model_type == 'rf':
        model = Golem(goal='max', ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',n_jobs=1)

    y_3 = []
    if X_samples.all != None:
        X_3 = X_samples
        n_samples = X_3.shape[0]
        y_3 = y_samples
        
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
        y_3 = []
        for x in X_3:
            y_3.append(objective_function(x, uncertainty=0))  
    
    y_3 =np.array(y_3)  
    y_dim = len(y_samples_RF.shape)  # 目标空间的维度    

    mu_wG = []
    std_w = []
    std_WG = []
     
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)
        mu_wG = []
        std_w = []
        std_WG = []

        # kriging 推荐值
        x_next_3 = propose_location(gaussian_ei, model,model_RF,X_3, y_3, bounds, n_restarts = 10,distributions = dists)
        # x_next_3 = propose_location(gaussian_ei, model,model_RF,X_3, y_3, bounds, n_restarts = 10)
        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        y_next_3 = objective_function(x_next_3.flatten()).reshape(y_dim,)
        # X_3 = np.vstack([X_3, x_next_3])
        # y_3.append(y_next_3)
        print(x_next_3)
        print(y_next_3)
        # _,y_pred_std_3 = model.predict(x_true_3,return_std=True)

    if model_type == 'rf':
        dists = distributions
        model_RF.fit(X_3, y_3)
        model.fit(X=X_3, y=y_3)


        # RandomForest 推荐值
        x_next_3 = propose_location(model,model_RF,X_3, y_3, bounds,distributions = dists,model_type=model_type, n_restarts = 10)
        y_next_3 = objective_function(ini_param,x_next_3.flatten())[0][0].reshape(y_dim,)
        
        # y_3.append(y_next_3)
        print(x_next_3.flatten())
        print(y_next_3)
   
        
    return x_next_3.flatten(),y_next_3,model,model_RF#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3

if __name__ == '__main__': 
    # samples=np.array([[0], [0.22], [0.39], [0.63], [0.86],[1]])
    objective_function = microalgae_growth
    space2=Space([(26,104),(0.335,0.410),(0.0064,0.025),(12,24)]) #N0[335,385]mg/L,I0[26,78]umol/s.m^2,tf[12,24]h,P0[6.4,25]mg L-1
    D = 4
    N_TRIALS = 1
    n_samples = 10
    n_iterations = 40

    bounds = np.array(space2.bounds).T
    bounds = bounds.tolist()
    dists = [DiscreteLaplace(2),Normal(0.012),Normal(0.001),DiscreteLaplace(0.1)] # 第一个维度上是t，但是t只参与预测，不参与优化

    
    
    
    for trial in range(1, N_TRIALS + 1):
        
        # samples = generate_sample_points(bounds, n_samples=n_samples,D=D)
        # arr = pd.read_excel('2D_data.xlsx')
        X_samples = train_x_curve
        y_samples = train_y_curve
        X_samples_RF = X_samples
        y_samples_RF = y_samples
        X_samples_GP = X_samples
        y_samples_GP = y_samples
        model_type = 'rf'
        
        X_samples_GP_list = []
        X_samples_RF_list = []
        X_samples_GP_list_all = []
        X_samples_RF_list_all = []
        
        y_samples_GP_list = []
        y_samples_RF_list = []
        y_samples_GP_list_all = []
        y_samples_RF_list_all = []
        
        best_observed_preference_all_RF = []
        best_observed_preference_all_GP = []
        
        
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        
        best_observed_preference_RF = []
        best_observed_value_RF = y_samples_RF.min()
        best_observed_preference_RF.append(best_observed_value_RF)
        
    
        best_observed_preference_GP = []
        best_observed_value_GP = y_samples_GP.min()
        best_observed_preference_GP.append(best_observed_value_GP)
        
        X_samples_GP_list = []
        X_samples_RF_list = []
        y_samples_GP_list = []
        y_samples_RF_list = []
        
        best_observed_preference_GP = []
        best_observed_value_GP = y_samples_GP.min()
        best_observed_preference_GP.append(best_observed_value_GP)
        for i in range(n_iterations):

            time_start = time.time()
            
            x_next_RF,y_next_RF,model3,model_RF = robust_optimization_GW_3( objective_function, bounds,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=D,X_samples=X_samples_RF,y_samples=y_samples_RF,model_type='rf',nproc=1)
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using random forest using %d 秒'%time_sum)
            # time_start = time.time()
            # x_next_GP,y_next_GP,model_GP,model_RF_2 = robust_optimization_GW_3( objective_function, bounds,
            #                                                                     n_samples, n_iterations,distributions=dists,
            #                                                                     D=D,samples=X_samples_GP,model_type='gp')
            # time_end = time.time()
            # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            # print('The programm that robust optimiztion using Gaussian process using %d 秒'%time_sum)
            X_samples_RF = np.vstack([X_samples_RF, x_next_RF])
            y_samples_RF = np.hstack([y_samples_RF, y_next_RF])  
            # X_samples_RF_list.append(X_samples_RF)
            # y_samples_RF_list.append(y_samples_RF)
            best_observed_preference_RF.append(max(best_observed_preference_RF[-1],y_next_RF))
            
            # X_samples_GP = np.vstack([X_samples_GP, x_next_GP])
            # y_samples_GP = np.hstack([y_samples_GP, y_next_GP])

            # best_observed_preference_GP.append(min(best_observed_preference_GP[-1],y_next_GP))
            

            


        best_observed_preference_all_RF.append(best_observed_preference_RF)  
        # X_samples_RF_list_all.append(X_samples_RF_list)
        # y_samples_RF_list_all.append(y_samples_RF_list)
        X_samples_RF_list_all.append(X_samples_RF)
        y_samples_RF_list_all.append(y_samples_RF)
