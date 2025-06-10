import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,poisson,dlaplace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from skopt.learning import RandomForestRegressor as RF_std
import pyximport
pyximport.install()
from golem import * 
import warnings
from scipy.optimize import minimize
from scipy.integrate import nquad
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal,DiscreteLaplace,Poisson
from plot_ro import plot_RO_K,plot_RO_RF,plot_RO_K_3,plot_RO_RF_3
from integrated import convolute_RF,convolute_K
# from integrated_mc import convolute_RF,convolute_K,convolute_RF_algae
import time
from microalgae_model.model3 import Logistic
from microalgae_model.cycle import Cycle
from skopt import Space
from skopt.space import Real, Integer, Categorical
import random
import pandas as pd
from sklearn.model_selection import train_test_split

import os
path = os.getcwd()
datapaths = os.path.join(path,'algae_curve_data_7.xlsx')
algae_data_curve_4 = pd.read_excel(datapaths)
algae_data_curve_4 = np.array(algae_data_curve_4)
train_idx = random.sample(range(0,len(algae_data_curve_4)),int(len(algae_data_curve_4)/2-1)) 

ini_param = [0.0138,121,0.06]
def microalgae_growth(ini_param,var,t_up=120,t_lb=0):
    
    parameter_light = {"xmax":1.8,  # 环境最大浓度  ,g/m3
                 "um":2.5946,   # 最大比增长率     ,1/h
                 "Ka":1.84, # 光衰减系数        ,m2/g
                 "KIi":477.882,    # 光抑制常数    3            ,μmol/m2 s
                 "KIs":300.7205 ,  # 光饱和常数    4           ,μmol/m2 s
                 "ud":0.0002,   # 比光合自养衰减率 5
                 "KNi":353.5633,    # N元素营养抑制常数  6       ,mg/L
                 "KNs":250.502,   # N元素营养饱和常数  7      ,mg/L
                 "Yx/N":15.871,   # N元素吸收速率 8            ,无量纲
                   "m/N":3.6401,     # N元素相关系数
                 #"alpha":750,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）9
                  #"beta":0.0005, # 产品(脂质)的具体形成速率 https://doi.org/10.1007/s10811-016-0841-4 
                   "KPi":30.1707 ,    # C元素营养抑制常数  9
                  "KPs":14.32927,   # C元素营养饱和常数  10
                   "Yx/P":18.678067,   # 乙酸吸收速率 18
                   "m/P":0.242953
                }
    parameter_dark= {"xmax":1.285,  # 环境最大浓度  ,g/m3
                     
                    "um":1.57287343607939,  # 氮吸收最大速率  0
                     "ud":0.005,   # 比光合自养衰减率 5
                 "KNi":373.31,    # 营养抑制常数  1
                  "KNs":130.22,   # 营养饱和常数  2
                 "Yx/N":5.06983 ,  # N元素吸收速率 3
                  "m/N":5.418317,     # N元素相关系数
                # "alpha":91.235,  # α是细胞生长引起的产物形成的瞬时产量系数（g g−1）4
                 #"beta":39.596, # 产品(脂质)的具体形成速率 10 https://doi.org/10.1007/s10811-016-0841-4 5
                  "KPi":28.65422 ,    # C元素营养抑制常数  9
                  "KPs":12.2133,   # C元素营养饱和常数  10
                  "Yx/P":3.0748,   # 乙酸吸收速率 13
                   "m/P":0.11
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
    return [X[-1],N[-1],P[-1]],[X1,N1,P1], [X[84],N[84],P[84]]

train_x_curve = algae_data_curve_4[:,1:5] 

train_x_curve = train_x_curve[train_idx]
train_y_curve_ = []
for x in train_x_curve:
    train_y_curve_.append(microalgae_growth(ini_param,x,t_up=120,t_lb=0)[2][0])
train_y_curve = np.array(train_y_curve_).flatten()
# train_y_curve = algae_data_curve_4[:,5]  

import geatpy as ea
from myproblem import MyProblem

def EI_optimize(model,model_RF,model_std,model_std_mu,X_sample,bounds,distributions,model_type,goal='min',uncertainty='GW'):
    
    
    problem = MyProblem(X_sample,model,model_RF,model_std,model_std_mu,bounds,distributions,model_type,goal=goal,uncertainty=uncertainty)
    algorithm = ea.soea_EGA_templet(problem,
                                ea.Population(Encoding='RI',NIND=20),
                                MAXGEN=400,  
                                logTras=0
                                )  
    # algorithm.mutOper.F = 0.5  
    # algorithm.recOper.XOVR = 0.7  
    res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False,)
    
    return res


def propose_location(model,model_RF,model_std,model_std_mu, X_sample, Y_sample, bounds,uncertainty='GW',distributions=None,model_type='gp', goal=None,n_restarts = 2):
    dim = X_sample.shape[1]   # X_sample: Sample locations (n x d). 所以dim = 1
    min_val = 1
    min_x = None
    
    EI_res = EI_optimize(model,model_RF,model_std,model_std_mu,X_sample,bounds,distributions,model_type,goal=goal,uncertainty=uncertainty)  
    min_x = EI_res["Vars"]        
            
    return min_x.reshape(1, -1)

def robust_optimization_GW_3(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,X_samples=None,y_samples=None,uncertainty='GW',model_type='gp',goal='min',nproc=1):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
        model_std = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        # model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error')
    if model_type == 'rf':
        model = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        # model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',n_jobs=nproc,min_variance=0.01)
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True) 
    # model_RF = RF_std()
    # model = GaussianProcessRegressor()
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
        
        # y_3 =np.array(y_3)     
    # if X_3.shape[1] != D:
    #     raise ValueError("The dimansion of X must equal to D")
    
    y_3 =np.array(y_3)  
    y_dim = len(y_samples_RF.shape)  # 目标空间的维度    

    mu_wG = []
    std_w = []
    std_WG = []
     
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)
        x_next_3 = propose_location( model,model_RF,X_3, y_3, bounds,uncertainty=uncertainty, n_restarts = 20,distributions = dists)

        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        y_next_3 = objective_function(x_next_3.flatten()).reshape(y_dim,)

        print(x_next_3)
        print(y_next_3)
        # _,y_pred_std_3 = model.predict(x_true_3,return_std=True)

    if model_type == 'rf':
        dists = distributions
        model_RF.fit(X_3, y_3.ravel())
        model.fit(X_3, y_3)
        model_std_mu.fit(X_3,y_3)
        std_G = model_RF.predict(X_3,return_std=True)[1]
        model_std.fit(X=X_3, y=std_G)

        x_next_3 = propose_location( model,model_RF,model_std,model_std_mu,X_3, y_3, bounds,uncertainty=uncertainty,distributions = dists,model_type=model_type,goal=goal, n_restarts = 10)
        print(x_next_3)
        print(x_next_3.shape)
    
        x_next_3[0][0] = x_next_3[0][0] -poisson.rvs(mu=5, size=1)
        x_next_3[0][1] = x_next_3[0][1] + norm.rvs(0, 30, 1)
        x_next_3[0][2] = x_next_3[0][2]+ norm.rvs(0, 10, 1)
        x_next_3[0][3] = x_next_3[0][3]#- 0.05*(poisson.rvs(mu=3, size=1))
       
        y_X,_ ,y_next_3 = objective_function(ini_param=ini_param,var=x_next_3.flatten())      
    return x_next_3,y_next_3[0],model,model_RF,model_std,model_std_mu#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3

#==================================== no noise in the iteration ================================#
def robust_optimization_GW_3_no(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,X_samples=None,y_samples=None,uncertainty='GW',model_type='gp',goal='min',nproc=1):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
        model_std = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        # model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error')
    if model_type == 'rf':
        model = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std = Golem(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        # model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',n_jobs=nproc,min_variance=0.01)

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

    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)

        x_next_3 = propose_location( model,model_RF,X_3, y_3, bounds,uncertainty=uncertainty, n_restarts = 20,distributions = dists)
        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        y_next_3 = objective_function(x_next_3.flatten()).reshape(y_dim,)

        print(x_next_3)
        print(y_next_3)


    if model_type == 'rf':
        dists = distributions
        model_RF.fit(X_3, y_3.ravel())
        model.fit(X_3, y_3)
        model_std_mu.fit(X_3,y_3)

        std_G = model_RF.predict(X_3,return_std=True)[1]
        model_std.fit(X=X_3, y=std_G)
        



        # RandomForest 推荐值
        x_next_3 = propose_location( model,model_RF,model_std,model_std_mu,X_3, y_3, bounds,uncertainty=uncertainty,distributions = dists,model_type=model_type,goal=goal, n_restarts = 10)
        
        y_X,_ ,y_next_3 = objective_function(ini_param=ini_param,var=x_next_3.flatten())
       
        
    return x_next_3,y_next_3[0],model,model_RF,model_std,model_std_mu#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3


def min_index(data): 
    index = []  
    # data = data.A  
    dim_1 = data.ravel()  
    min_n = min(dim_1)  
    for i in range(len(dim_1)):
        if dim_1[i] == min_n:  
            pos = np.unravel_index(i, data.shape, order='C') 
            index.append(pos)  
    return np.array(index)


if __name__ == '__main__': 
    # samples=np.array([[0], [0.22], [0.39], [0.63], [0.86],[1]])
    objective_function = microalgae_growth
    space2=Space([(26,578),(335,585),(6.4,55),(12,24)]) #N0[335,385]mg/L,I0[26,78]umol/s.m^2,tf[12,24]h,P0[6.4,25]mg L-1
    D = 4
    N_TRIALS = 10
    n_samples = 10
    n_iterations = 40
    model_type = 'rf'
    bounds = np.array(space2.bounds).T
    bounds = bounds.tolist()
    dists = [Poisson(2),Normal(30),Normal(2),Delta()] 
    dists_no = [Delta(),Delta(),Delta(),Delta()]
    
    # 随机森林无不确定性
    X_samples_RF_all_no = []
    y_samples_RF_all_no = []
    # 随机森林无不确定性+噪声
    X_samples_RF_all = []
    y_samples_RF_all = []
    # 随机森林输入不确定性+噪声
    X_samples_RF_W_all = []
    y_samples_RF_W_all = []
    # 随机森林双重不确定性+噪声
    X_samples_RF_GW_all = []
    y_samples_RF_GW_all = []

    best_observed_preference_all_RF_no = []
    best_observed_preference_all_RF = []
    best_observed_preference_all_RF_W = []
    best_observed_preference_all_RF_GW = []

    for trial in range(1, N_TRIALS + 1):
        
        # samples = generate_sample_points(bounds, n_samples=n_samples,D=D)
        # arr = pd.read_excel('2D_data.xlsx')
        X_samples = train_x_curve
        y_samples = train_y_curve

        # 随机森林无不确定性
        X_samples_RF_no = X_samples
        y_samples_RF_no = y_samples
        # 随机森林无不确定性+噪声
        X_samples_RF = X_samples
        y_samples_RF = y_samples
        # 随机森林输入不确定性+噪声
        X_samples_RF_W = X_samples
        y_samples_RF_W = y_samples
        # 随机森林双重不确定性+噪声
        X_samples_RF_GW = X_samples
        y_samples_RF_GW = y_samples
        
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        
        # 随机森林无不确定性
        best_observed_preference_RF_no = []
        best_observed_value_RF_no = y_samples_RF_no.max()
        best_observed_preference_RF_no.append(best_observed_value_RF_no)
        # 随机森林无不确定性+噪声
        best_observed_preference_RF = []
        best_observed_value_RF = y_samples_RF.max()
        best_observed_preference_RF.append(best_observed_value_RF)
        # 随机森林输入不确定性+噪声
        best_observed_preference_RF_W = []
        best_observed_value_RF_W = y_samples_RF_W.max()
        best_observed_preference_RF_W.append(best_observed_value_RF_W)
        # 随机森林双重不确定性+噪声
        best_observed_preference_RF_GW = []
        best_observed_value_RF_GW = y_samples_RF_GW.max()
        best_observed_preference_RF_GW.append(best_observed_value_RF_GW)
        
        
        
        X_samples_GP_list_no = []
        X_samples_RF_list_no = []
        y_samples_GP_list_no = []
        y_samples_RF_list_no = []
        
        # best_observed_preference_GP = []
        # best_observed_value_GP = y_samples_GP.min()
        # best_observed_preference_GP.append(best_observed_value_GP)
        # 有考虑不确定性情况的优化
        for i in range(n_iterations):
            print(f"Trial: {trial}, Iteration: {i}")
            time_start = time.time()
            # 考虑双重不确定性的优化+noise
            x_next_RF_GW,y_next_RF_GW,model3_GW,model_RF_GW,model_std_GW,model_std_mu_GW = robust_optimization_GW_3( objective_function, bounds,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=D,X_samples=X_samples_RF,y_samples=y_samples_RF,model_type='rf',goal='max',nproc=1)
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using random forest GW using %d 秒 + noise'%time_sum)
            # 仅考虑输入不确定性的优化+noise
            time_start = time.time()
            x_next_RF_W,y_next_RF_W,model3_W,model_RF_W,model_std_W,model_std_mu_W = robust_optimization_GW_3( objective_function, bounds,
                                                                                n_samples, n_iterations,distributions=dists_no,
                                                                                D=D,X_samples=X_samples_RF,y_samples=y_samples_RF,uncertainty=None,model_type='rf',goal='max',nproc=1)
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using random forest W using %d 秒 + noise'%time_sum)
            # 不考虑输入不确定性的优化+noise
            time_start = time.time()
            x_next_RF,y_next_RF,model3,model_RF,model_std,model_std_mu = robust_optimization_GW_3( objective_function, bounds,
                                                                                n_samples, n_iterations,distributions=dists_no,
                                                                                D=D,X_samples=X_samples_RF,y_samples=y_samples_RF,uncertainty=None,model_type='rf',goal='max',nproc=1)
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using random forest EGO_noise using %d 秒 + noise'%time_sum)
            # 不考虑输入不确定性的优化
            time_start = time.time()
            x_next_RF_no,y_next_RF_no,model3_no,model_RF_no,model_std_no,model_std_mu_no = robust_optimization_GW_3_no( objective_function, bounds,
                                                                                n_samples, n_iterations,distributions=dists_no,
                                                                                D=D,X_samples=X_samples_RF,y_samples=y_samples_RF,uncertainty=None,model_type='rf',goal='max',nproc=1)
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using random forest EGO using %d 秒'%time_sum)
            
            X_samples_RF_GW = np.vstack([X_samples_RF_GW, x_next_RF_GW])
            y_samples_RF_GW = np.hstack([y_samples_RF_GW, y_next_RF_GW])
            X_samples_RF_W = np.vstack([X_samples_RF_W, x_next_RF_W])
            y_samples_RF_W = np.hstack([y_samples_RF_W, y_next_RF_W])
            X_samples_RF = np.vstack([X_samples_RF, x_next_RF])
            y_samples_RF = np.hstack([y_samples_RF, y_next_RF]) 
            X_samples_RF_no = np.vstack([X_samples_RF_no, x_next_RF_no])
            y_samples_RF_no = np.hstack([y_samples_RF_no, y_next_RF_no])  

            best_observed_preference_RF.append(max(best_observed_preference_RF[-1],y_next_RF))
            best_observed_preference_RF_no.append(max(best_observed_preference_RF_no[-1],y_next_RF_no))
            best_observed_preference_RF_GW.append(max(best_observed_preference_RF_GW[-1],y_next_RF_GW))
            best_observed_preference_RF_W.append(max(best_observed_preference_RF_W[-1],y_next_RF_W))

        

        
        best_observed_preference_all_RF_no.append(best_observed_preference_RF_no)  
        best_observed_preference_all_RF.append(best_observed_preference_RF)  
        best_observed_preference_all_RF_W.append(best_observed_preference_RF_W)  
        best_observed_preference_all_RF_GW.append(best_observed_preference_RF_GW)  

        # 随机森林无不确定性
        X_samples_RF_all_no.append(X_samples_RF_no)
        y_samples_RF_all_no.append(y_samples_RF_no)
        # 随机森林无不确定性+噪声
        X_samples_RF_all.append(X_samples_RF)
        y_samples_RF_all.append(y_samples_RF)
        # 随机森林输入不确定性+噪声
        X_samples_RF_W_all.append(X_samples_RF_W)
        y_samples_RF_W_all.append(y_samples_RF_W)
        # 随机森林双重不确定性+噪声
        X_samples_RF_GW_all.append(X_samples_RF_GW)
        y_samples_RF_GW_all.append(y_samples_RF_GW)

        print(X_samples_RF_GW)
        print(best_observed_preference_RF_GW)
        print(X_samples_RF_W)
        print(best_observed_preference_RF_W)
        print(X_samples_RF)
        print(best_observed_preference_RF)
        print(X_samples_RF_no)
        print(best_observed_preference_RF_no)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        folder_name = "algae" + f"output_{current_time}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # np.savetxt(folder_name+'/X_samples_RF.txt', X_samples_RF, delimiter=',')
        # np.savetxt(folder_name+'/y_samples_RF.txt', y_samples_RF, delimiter=',')
        # np.savetxt(folder_name+'/best_observed_preference_RF.txt', best_observed_preference_RF, delimiter=',')
        # np.savetxt(folder_name+'/X_samples_RF_no.txt', X_samples_RF_no, delimiter=',')
        # np.savetxt(folder_name+'/y_samples_RF_no.txt', y_samples_RF_no, delimiter=',')
        # np.savetxt(folder_name+'/best_observed_preference_RF_no.txt', best_observed_preference_RF_no, delimiter=',')
        df01 = pd.DataFrame(best_observed_preference_RF_GW)
        df02 = pd.DataFrame(best_observed_preference_RF_W)
        df03 = pd.DataFrame(best_observed_preference_RF)
        df04 = pd.DataFrame(best_observed_preference_RF_no)

        X_dimension=np.array(X_samples_RF_GW).shape
        Y_dimension=np.array(y_samples_RF_GW).shape
        print(np.array(X_samples_RF_GW))
        print(np.array(X_samples_RF_GW).shape)
        # df05 = pd.DataFrame(np.array(X_samples_RF_GW_all).reshape(np.array(X_samples_RF_GW_all).shape[1],np.array(X_samples_RF_GW_all).shape[2]))
        # df05 = pd.DataFrame(np.array(X_samples_RF_GW_all).reshape(n_iterations*X_dimension[1],X_dimension[2]))
        df05 = pd.DataFrame(X_samples_RF_GW)
        df06 = pd.DataFrame(y_samples_RF_GW)
        df07 = pd.DataFrame(X_samples_RF_W)
        df08 = pd.DataFrame(y_samples_RF_W)
        df09 = pd.DataFrame(X_samples_RF)
        df10 = pd.DataFrame(y_samples_RF)
        df11 = pd.DataFrame(X_samples_RF_no)
        df12 = pd.DataFrame(y_samples_RF_no)
        df01.to_excel(folder_name+'/best_observed_preference_RF_GW.xlsx')
        df02.to_excel(folder_name+'/best_observed_preference_RF_W.xlsx')
        df03.to_excel(folder_name+'/best_observed_preference_RF.xlsx')
        df04.to_excel(folder_name+'/best_observed_preference_RF_no.xlsx')
        df05.to_excel(folder_name+'/X_samples_RF_GW.xlsx')
        df06.to_excel(folder_name+'/y_samples_RF_GW.xlsx')
        df07.to_excel(folder_name+'/X_samples_RF_W.xlsx')
        df08.to_excel(folder_name+'/y_samples_RF_W.xlsx')
        df09.to_excel(folder_name+'/X_samples_RF.xlsx')
        df10.to_excel(folder_name+'/y_samples_RF.xlsx')
        df11.to_excel(folder_name+'/X_samples_RF_no.xlsx')
        df12.to_excel(folder_name+'/y_samples_RF_no.xlsx')

        
