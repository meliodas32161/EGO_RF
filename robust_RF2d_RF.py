import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from skopt.learning import RandomForestRegressor as RF_std
import pyximport
from golem import * 
import warnings
from scipy.optimize import minimize
from scipy.integrate import nquad
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal,BoundedUniform,Uniform
from plot_ro import plot_RO_K,plot_RO_RF,plot_RO_K_3,plot_RO_RF_3
from integrated import convolute_RF,convolute_K
# from integrated_mc import convolute_RF,convolute_K
import time
# from numba import jit
import multiprocessing as mp 
import os
import pandas as pd
from joblib import Parallel, delayed
def generate_sample_points(bounds, n_samples,D=1):
    """Generate Latin hypercube sample points within the given bounds."""
    if type(bounds) == list:
        bounds = np.array(bounds)
    d = 1.0 / n_samples 
    samples = np.empty([n_samples, D])
    temp = np.empty([n_samples])
    for i in range(D):
        # 维度迭代
        for j in range(n_samples):
            # 根据采样数量在每个维度进行分层（=n_sample），每个层都要进行采样
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp) # 随机打乱顺序
        for j in range(n_samples):
            samples[j, i] = temp[j]
    if np.any(bounds[0]> bounds[1]):
        print('Range error')
        return None
    # multiply 求两个矩阵的内积
    # add 两个矩阵相加
    np.add(np.multiply(samples, (bounds[1] - bounds[0]), out=samples), bounds[0], out=samples)
    
    return samples

# 一维真实函数
def onedimention_problem(x, uncertainty=0):
    """onedimention_problem function with input uncertainty."""
    return (6*x-2)**2 * np.sin(12*x-4)+ 8*x

# 二维真实函数 Concurrent treatment of parametric uncertainty and metamodeling uncertainty in robust design
def Twodimention_problem(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    return 1.9*(1.35+np.exp(x[0])*np.sin(7*x[0])*13*(x[0]-0.6)**2*np.exp(-x[1])*np.sin(7*x[1]))

# 二维真实函数——cliff
def Twodimention_cliff(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    obj = 0
    for i in range(len(x)):
        obj = obj+10 / (1+0.3*np.exp(6*x[i])) + 0.2*x[i]**2
    # obj = np.array([obj + 10 / (1+0.3*np.exp(6*x[i])) + 0.2*x[i]**2 for i in range(len(x))])
    return obj
# 二维真实函数——Bertsimas
def Twodimention_Bertsimas(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    f1 = -2*x[0]**6 + 12.2*x[0]**5 - 21.2*x[0]**4 - 6.4*x[0]**3 + 4.7*x[0]**2- 6.2*x[0]   
    f2 = x[1]**6 - 11*x[1]**5 + 43.3*x[1]**4 - 10*x[1] - 74.8*x[1]**3 + 56.9*x[1]**2
    f3 = -4.1*x[0]*x[1] - 0.1*x[0]**2*x[1]**2 + 0.4*x[0]*x[1]**2 + 0.4*x[0]**2*x[1]
    
    return f1+f2-f3
# 二维真实函数 Robust expected improvement for Bayesian optimization
def Twodimention_problem_2(x, uncertainty=0):
    f1 = -2*x[0]**6 + 12.2*x[0]**5-21.2*x[0]**4 + 6.4*x[0]**3 + 4.7*x[0]**2 - 6.2*x[0]
    f2 = -1*x[1]**6 + 11*x[1]**5 - 43.3*x[1]**4 + 74.8*x[1]**3 - 56.9*x[1]**2 + 10*x[1]
    f3 = 4.1*x[0]*x[1] + 0.1*x[0]**2*x[1]**2 - 0.4*x[0]*x[1]**2 - 0.4*x[0]**2*x[1]
    return -(f1 + f2 + f3)
# 三维真实函数——two-dimensional Branin function  + one-dimensional test problem
def Threedimention_cliff(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    f1 = x[1] - 5.1/(4*np.pi)*x[1] + 5/np.pi*x[0] -6 
    f2 = 10*((1 - 1/(8*np.pi)) * np.cos(x[0]) + 1)
    f3 = (6*x[2] - 2)**2 * np.sin(12*x[2] - 4) + 8*x[2]
    
    return f1+f2+f3
    

# 负的真实函数
def rosenbrock_(x, uncertainty=0):
    """Rosenbrock function with input uncertainty."""
    return -onedimention_problem(x,uncertainty)

# 真实函数的鲁棒对应问题
def F(x,e=0,n_restarts = 15):
    bound = [x[0],x[-1]]
    dim = x.shape[1]
    # bounds = np.array([[-2-e,2+e]])
    # min_val = 10
    F_ = []
    for xx in x:
        min_val = float ('inf')
        bounds = np.array([[xx-e/2,xx+e/2]])
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            constraints = ({'type': 'ineq', 'fun':lambda X0: X0 - e - bound[0]},
                           {'type': 'ineq', 'fun':lambda X0: bound[0]-X0 - e})
            res = minimize(fun=rosenbrock_,x0=x0,method='L-BFGS-B',bounds=bounds,tol=1e-8)
            
            if res.fun < min_val:
                min_val = res.fun
                # xmax = res.x
        F_.append(min_val)
            
    return np.array(F_,dtype=object)



# EI准则

def gaussian_ei(X,*args):
    
    xi = 0.1
    n_restarts=20
    model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,y_,goal = args
    n = X.shape[0] 
    

    if model_type == 'gp':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            mu,_,std = convolute_K(X,model,bounds,dists=distribution)
            if (mu.ndim != 1):
                mu = mu.flatten()
            
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                              "however both must be 1-dimensional. Did you train "
                              "your model with an (N, 1) vector instead of an "
                              "(N,) vector?"
                              .format(mu.ndim, std.ndim))
    elif model_type == 'rf':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,model_std_mu=model_std_mu,bound=bounds)
            if (mu.ndim != 1):
                mu = mu.flatten()
            if (std.ndim != 1):
                std = std.flatten()
            
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                              "however both must be 1-dimensional. Did you train "
                              "your model with an (N, 1) vector instead of an "
                              "(N,) vector?"
                              .format(mu.ndim, std.ndim))

    values = np.zeros_like(mu)
    # values  = np.zeros_like(mu)
    mask = std > 0
    
    # y_opt = 0 # np.max(mu)
    y_opt = y_
    # print(mu,std)
    # if goal == 'min':
    #     improve = y_opt-(mu[mask]+2*std[mask])- xi
    # else:
    #     improve = (mu[mask]-2*std[mask])-y_opt- xi
    if goal == 'min':
        improve = y_opt-mu[mask]- xi
    else:
        improve = mu[mask]-y_opt- xi
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore
    return values

def gaussian_ei_GW(X,*args):
    
    xi = 0.1
    n_restarts=20
    X_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,uncertainty,goal = args
    n = X.shape[0] 
    

    if model_type=='gp':
        if uncertainty == 'GW':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainty)     
            mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainty)
            if (mu.ndim != 1):
                mu = mu.flatten()
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainty == 'W':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainty)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainty)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainty == None:
            mu_samples,std_samples = model.predict(X_sample,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,std = model.predict(X,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                    
    elif model_type == 'rf':
        if uncertainty == 'GW':
            mu_samples,_,std_samples = convolute_RF(X_sample,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
                                                    model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainty)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
                                        model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainty)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainty == 'W':
            mu_samples,std_samples,var_samples = model.predict(X_sample,distributions=distribution,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std,var = model.predict(X,distributions=distribution,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainty == None:
            mu_samples,std_samples = model_RF.predict(X_sample,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std = model_RF.predict(X,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
            
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                              "however both must be 1-dimensional. Did you train "
                              "your model with an (N, 1) vector instead of an "
                              "(N,) vector?"
                              .format(mu.ndim, std.ndim))

    if goal=='min':
        y = mu_samples + 2*std_samples
        y_ = np.min(y)
    if goal=='max':
        y = mu_samples - 2*std_samples
        y_ = np.max(y)
    values = np.zeros_like(mu)
    # values  = np.zeros_like(mu)
    mask = std > 0
    
    y_opt = y_
    # print(mu,std)
    if goal == 'min':
        improve = y_opt-(mu[mask])- xi
    else:
        improve = (mu[mask])-y_opt- xi
    # if goal == 'min':
    #     improve = y_opt-mu[mask]- xi
    # else:
    #     improve = mu[mask]-y_opt- xi
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore
    # values[std == 0.0] = 0.0
    return values

def gaussian_lcb_GW(X,*args):
    kappa = 2.576
    xi = 0.1
    n_restarts=20
    X_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,uncertainty,goal = args
    n = X.shape[0] 
    

    if model_type=='gp':
        if uncertainty == 'GW':
            # mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainty)     
            mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainty)
            if (mu.ndim != 1):
                mu = mu.flatten()
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainty == 'W':
            # mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainty)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainty)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
                
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainty == None:
            # mu_samples,std_samples = model.predict(X_sample,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,std = model.predict(X,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                    
    elif model_type == 'rf':
        if uncertainty == 'GW':
            # mu_samples,_,std_samples = convolute_RF(X_sample,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
            #                                         model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainty)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
                                        model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainty)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainty == 'W':
            # mu_samples,std_samples,var_samples = model.predict(X_sample,distributions=distribution,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std,var = model.predict(X,distributions=distribution,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainty == None:
            # mu_samples,std_samples = model_RF.predict(X_sample,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std = model_RF.predict(X,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
            
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                              "however both must be 1-dimensional. Did you train "
                              "your model with an (N, 1) vector instead of an "
                              "(N,) vector?"
                              .format(mu.ndim, std.ndim))


    values = np.zeros_like(mu)
    # values  = np.zeros_like(mu)
    mask = std > 0

    values[mask] = mu[mask] + kappa*std[mask] 
    # values[std == 0.0] = 0.0
    return values

def parallel_optimization(min_obj, bounds, bound, arg, n_restarts, dim, n_jobs=-1):
    """
    并行运行多个优化任务，从不同的初始点开始
    
    Parameters:
        min_obj: 目标函数
        bounds: 初始点采样范围，如 [(low, high), (low, high), ...]
        bound: scipy.optimize 的 bounds 格式
        arg: 传递给 min_obj 的额外参数
        n_restarts: 随机初始点数量
        dim: 变量维度
        n_jobs: 并行任务数（-1 表示使用所有 CPU 核心）
        
    Returns:
        min_x: 最优解
        min_val: 最优值
    """
    # 生成所有初始点
    x0_list = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_restarts, dim)
    )

    # 定义单次优化任务
    def run_single_optimization(x0):
        res = minimize(min_obj, x0=x0, bounds=bound, args=arg, method='L-BFGS-B')
        return res.fun, res.x

    # 并行运行所有优化
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_optimization)(x0)
        for x0 in tqdm(x0_list, desc="Optimizing")  # 可选：显示进度条
    )
    return results

    # # 提取最优解
    # min_val, min_x = min(results, key=lambda x: x[0])
    # return min_x.reshape(1, -1)

# 定义采集函数取最大的函数

def propose_location(acquisition ,X_sample, Y_sample, bounds,model=None,model_RF=None,model_std=None,model_std_mu=None, 
                     distributions=None,model_type='gp',n_restarts = 10,uncertainty='GW',goal='min'):
    
    dim = X_sample.shape[1]   # X_sample: Sample locations (n x d). 所以dim = 1
    min_val = np.inf
    min_x = None
    # if uncertainty == 'GW' and model_type=='gp':
    #     mu_samples,_,std = convolute_K(X_sample,model=model,bound=bounds,dists=distributions)
    # if uncertainty == 'GW' and model_type=='rf':
    #     mu_samples,_,std = convolute_RF(X_sample,dists=distributions,model=model,
    #                                     model_RF=model_RF,model_std=model_std,model_std_mu=model_std_mu,bound=bounds)
    # if uncertainty == 'W'and model_type=='rf':
    #     mu_samples,std = convolute_RF_w(X_sample,dists=distributions,model=model,
    #                                     model_RF=model_RF,model_std=model_std,model_std_mu=model_std_mu,bound=bounds)    
    def min_obj(X,*args):
        # Minimization objective is the negative acquisition function
        # return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)#.reshape(-1, dim)
        return -acquisition(X,*args)#.reshape(-1, dim)
   
    # # Find the best optimum by starting from n_restart different random points.
    # if goal=='min':
    #     y = mu_samples + 2*std
    #     y_ = np.min(y)
    # if goal=='max':
    #     y = mu_samples - 2*std
    #     y_ = np.max(y)
    # y = mu_samples
    # if goal == 'max':
    #     y_ = np.max(y)
    # if goal == 'min':
    #     y_ = np.min(y)
    arg = (X_sample,model,model_RF,model_std,model_std_mu,bounds,distributions,model_type,uncertainty,goal)
    # 一维
    if dim == 1:       
        bound =np.array([bounds])
    else:
    # 二维以上
        bound = np.array(bounds).T
    
    for x0 in np.random.uniform(bounds[0], bounds[1], size=(n_restarts, dim)):
        
        res = minimize(min_obj, x0=x0, bounds=bound,args=arg,method = 'L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x   
    # results = parallel_optimization(
    # min_obj=my_objective_function,
    # bounds=[(0, 1), (-1, 1)],  # 初始点采样范围
    # bound=[(0, 1), (-1, 1)],    # scipy.optimize 的 bounds
    # arg=(additional_args,),      # 额外参数
    # n_restarts=100,              # 随机初始点数量
    # dim=2,                       # 变量维度
    # n_jobs=-1                    # 使用所有 CPU 核心
    # )
            
    return min_x.reshape(1, -1)

# 定义变量变换
def transform_to_t(x):
    x1, x2 = x
    t1 = (x1 + 0.95) / 4.15
    t2 = (x2 + 0.45) / 4.85
    return t1, t2
 
def transform_to_x(t):
    t1, t2 = t
    x1 = 4.15 * t1 - 0.95
    x2 = 4.85 * t2 - 0.45
    return x1, x2
 
# 定义变换后的目标函数
def transformed_function(t):
    # t是归一化后的变量，所以要现转化为原本的变量
    x1, x2 = transform_to_x(t)
    return Twodimention_problem_2([x1, x2])


# =============================高维考虑模型和参数不确定性的代理鲁棒优化=======================

def robust_optimization_GW_3(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,samples=None,y_samples=None,uncertainty = 'GW',model_type='gp',goal='min',nproc=1):
   
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
        # model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=nproc) # 前面是用model_RF预测的方差作为训练数据的用于模型不确定性的积分，后面注释是用golem融合了模型不确定性公式
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=nproc)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',n_jobs=nproc,min_variance=0.01)
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True) 
    # model_RF = RF_std()
    # model = GaussianProcessRegressor()

    X_3 = samples    
    # y_3 = []
    # n_samples = X_3.shape[0]
    # for x in X_3:
    #     #xi_ = transform_to_x(x)
    #     y_3.append(objective_function(x))
      
    # if X_3.shape[1] != D:
    #     raise ValueError("The dimansion of X must equal to D")
    
    # y_3 =np.array(y_3)  

    y_3 = y_samples
    
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)
        # 以下模型没啥用，为了输出一个训练的模型，所以就训了
        # model_RF.fit(X_3, y_3.ravel())
        # model_std.fit(X_3,y_3)
        # model_std_mu.fit(X_3,y_3)

        # kriging 推荐值
        # x_next_3 = propose_location(gaussian_ei, model,model_RF,X_3, y_3, bounds, n_restarts = 20,distributions = dists)
        x_next_3 = propose_location(gaussian_ei_GW, X_3, y_3, bounds,model=model, n_restarts = 10,distributions = dists,uncertainty=uncertainty,goal=goal)
        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        #xi_ = transform_to_x(x_next_3.flatten())  
        y_next_3 = objective_function(x_next_3.flatten())#.reshape(D,)
        y_next_3 = np.array(y_next_3)
        # X_3 = np.vstack([X_3, x_next_3])
        # y_3.append(y_next_3)
        print(x_next_3)
        print(y_next_3)
        # _,y_pred_std_3 = model.predict(x_true_3,return_std=True)

    if model_type == 'rf':
        dists = distributions
        model_RF.fit(X_3, y_3.ravel())
        model.fit(X_3, y_3)
        # model_std.fit(X_3,y_3)
        model_std_mu.fit(X_3,y_3)

        
        
        # 这一步是想用golem模型拟合一个用于预测输入x与方差的函数
        std_G = model_RF.predict(X_3,return_std=True)[1]
        model_std.fit(X=X_3, y=std_G)
        # Plot the current RF model
        # y_pred_mean_3 = model.forest.predict(x_true_3)
        # y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)


        # RandomForest 推荐值
        x_next_3 = propose_location(gaussian_ei_GW, X_3, y_3, bounds,model,model_RF,model_std,model_std_mu,distributions = dists,model_type=model_type, n_restarts = 10,uncertainty=uncertainty,goal=goal)
        # y_next_mean_3 = model.forest.predict(x_next_3)
        # y_next_3 = objective_function(x_next_3.flatten())
        # x_next_3_=np.zeros_like(x_next_3)
        # x_next_3_[0] = x_next_3.flatten()[0]*(bounds[1][0]-bounds[0][0])+bounds[0][0]
        # x_next_3_[1] = x_next_3.flatten()[1]*(bounds[1][1]-bounds[0][1])+bounds[0][1]
        x_next_3 = x_next_3.flatten()

        
        y_next_3 = objective_function(x_next_3)
        y_next_3 = np.array(y_next_3)
        # y_3.append(y_next_3)
        print(x_next_3)
        print(y_next_3)
        # plt.figure()
        # y_ei = gaussian_ei(x_true_3,model,model_RF,bounds,dists,model_type)
        # plt.plot(x_true_3.flatten(),y_ei)
        # plt.show()
        
        # _,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)
   
        
    return x_next_3,y_next_3,model,model_RF,model_std,model_std_mu#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3
def min_index(data): # 寻找最小值的所有索引
    index = []  # 创建列表,存放最小值的索引
    # data = data.A  # 若data是矩阵，需先转为array,因为矩阵ravel后仍是二维的形式，影响if环节
    dim_1 = data.ravel()  # 转为一维数组
    min_n = min(dim_1)  # 最小值max_n
    for i in range(len(dim_1)):
        if dim_1[i] == min_n:  # 遍历寻找最大值，并全部索引值进行操作
            pos = np.unravel_index(i, data.shape, order='C')  # 返回一维索引对应的多维索引，譬如4把转为(1,1),即一维中的第5个元素对应在二维里的位置是2行2列
            index.append(pos)  # 存入index
    return np.array(index)
#==================并行化处理===============#
def process_inner_loop(i, x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds,uncertainty):  
    mu_RF = []  
    sigma_RF = []  
    for j in range(len(y)):  
        mu_, _, sigma_ = convolute_RF(np.array([x[i], y[j]]), dists, model_3, model_RF, model_std,model_std_mu, bounds,uncertainty=uncertainty)  
        mu_RF.append(mu_.flatten())  
        sigma_RF.append(sigma_.flatten())  
    return mu_RF, sigma_RF  
def process_inner_loop_GP(i, x, y, dists, model_3,  bounds):  
    mu_GP = []  
    sigma_GP = []  
    for j in range(len(y)):  
        mu_, _, sigma_ = convolute_K(np.array([x[i], y[j]]), dists=dists, model=model_3, bound=bounds)  
        mu_GP.append(mu_.flatten())  
        sigma_GP.append(sigma_.flatten())  
    return mu_GP, sigma_GP  

def process_inner_loop_non(i, x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds):  
    mu_RF = []  
    sigma_RF = []  
    for j in range(len(y)):  
        mu_ = model_RF.predict(np.array([x[i], y[j]]).reshape(1,-1))  
        mu_RF.append(mu_.flatten())  
        # sigma_RF.append(sigma_.flatten())  
    return mu_RF #, sigma_RF  
def parallel_processing(x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds,fun_RF, num_workers=mp.cpu_count(),fun_name = 'process_inner_loop',uncertainty = 'GW'):  
    
    if  uncertainty ==  'GW':    
        mu_wG_RF = []  
        sigma_wG_RF = []  
        args = []
        for i in range(len(x)):
            args_i = (i, x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds,uncertainty)
            args.append(args_i)
        with mp.Pool(processes=num_workers) as pool:  
            results = pool.starmap(fun_RF,[a for a in args ])  
        for result in results:  
            mu_RF, sigma_RF = result  
            mu_wG_RF.append(mu_RF)  
            sigma_wG_RF.append(sigma_RF)  
        
        mu_wG_RF = np.array(mu_wG_RF)  
        sigma_wG_RF = np.array(sigma_wG_RF)   
        return mu_wG_RF, sigma_wG_RF  
    if  uncertainty ==  'W':
        mu_w_RF = []  
        sigma_w_RF = []  
        args = []
        for i in range(len(x)):
            args_i = (i, x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds,uncertainty)
            args.append(args_i)
        with mp.Pool(processes=num_workers) as pool:  
            results = pool.starmap(fun_RF,[a for a in args ])  
        for result in results:  
            mu_RF, sigma_RF = result  
            mu_w_RF.append(mu_RF)  
            sigma_w_RF.append(sigma_RF)  
        
        mu_w_RF = np.array(mu_w_RF)  
        sigma_w_RF = np.array(sigma_w_RF)   
        return mu_w_RF, sigma_w_RF  
def parallel_processing_GP(x, y, dists, model_3, bounds,fun_GP, num_workers=mp.cpu_count(),fun_name = 'process_inner_loop_GP'):  
    args = []
    for i in range(len(x)):
        args_i = (i, x, y, dists, model_3, bounds)
        args.append(args_i)
    with mp.Pool(processes=num_workers) as pool:  
        results = pool.starmap(fun_GP,[a for a in args ])  
    if  fun_name ==  'process_inner_loop_GP':    
        mu_wG_GP = []  
        sigma_wG_GP = []  
        for result in results:  
            mu_GP, sigma_GP = result  
            mu_wG_GP.append(mu_GP)  
            sigma_wG_GP.append(sigma_GP)  
        
        mu_wG_GP = np.array(mu_wG_GP)  
        sigma_wG_GP = np.array(sigma_wG_GP)   
        return mu_wG_GP, sigma_wG_GP  
    else:
        mu_GP = []  
        # sigma_wG_RF = []  
        for result in results:  
            mu_ = result  
            mu_GP.append(mu_)  
            # sigma_wG_RF.append(sigma_RF)  
        
        mu_GP = np.array(mu_GP)  
        # sigma_wG_RF = np.array(sigma_wG_RF)   
        return mu_GP#, sigma_wG_RF  
# regret评价指标
def Regretfunc(distribution,x_rm,model,model_RF,model_std,model_std_mu,bound,goal,model_type='rf',uncertainty='GW'):
    """
    objective: The objective function of oiptimization problem.For different obejective function
               the form of regret is different
    distribution:The distribution satisfied by design variables
    x_rm: The robust design variables generated by algoritm
    model:robust model
    """
    # if objective == 'Bertsimas':
    #     x_t = (0.2673, 0.2146)
    x_t = np.array([0.2673, 0.2146])
    # mu_wG_RF, sigma_wG_RF = parallel_processing(x, y, dists, model_3, model_RF, model_std,model_std_mu, bounds,fun=process_inner_loop,num_workers=mp.cpu_count(),fun_name ='process_inner_loop')
    if model_type == 'rf':
        regret_t_mu,_,regret_t_std = convolute_RF(x_t,distribution,model,model_RF,model_std,model_std_mu,bound,uncertainty=uncertainty)
        regret_rm_mu,_,regret_rm_std = convolute_RF(x_rm,distribution,model,model_RF,model_std,model_std_mu,bound,uncertainty=uncertainty)
    if model_type == 'gp':
        regret_t_mu,_,regret_t_std = convolute_K(x_t,model,bound,distribution,uncertainty=uncertainty)
        regret_rm_mu,_,regret_rm_std = convolute_K(x_rm,model,bound,distribution,uncertainty=uncertainty)
    if goal == 'min':
        regret_t = regret_t_mu+2*regret_t_std
        regret_rm = regret_rm_mu + 2*regret_rm_std
    if goal == 'max':
        regret_t = regret_t_mu - 2*regret_t_std
        regret_rm = regret_rm_mu - 2*regret_rm_std
    # regret = regret_rm-regret_t
    regret = regret_rm_mu - regret_t_mu
    regret2 = regret_rm - regret_t
    return regret,regret2
def distance(X_samples,goal,model_type='rf',uncertainty='GW'):
    """
    objective: The objective function of oiptimization problem.For different obejective function
               the form of regret is different
    distribution:The distribution satisfied by design variables
    x_rm: The robust design variables generated by algoritm
    model:robust model
    """
    # if objective == 'Bertsimas':
    #     x_t = (0.2673, 0.2146)
    # if len(X_samples[0]==1):
    x_t = np.array([0.198, 0.085])
    print(X_samples.shape)
    X_samples_ = np.array(transform_to_t(np.squeeze(X_samples.tolist()))) 
    distance = np.linalg.norm(x_t-X_samples_, ord=2)
    # if len(X_samples[0]==2):
    #     x_t = np.array([0.198, 0.085])
    #     x = transform_to_t(X_samples)
    #     distance = np.linalg.norm(x, ord=2)
    
    return distance
#%% 多维函数的实验
if __name__ == '__main__': 
    # samples=np.array([[0], [0.22], [0.39], [0.63], [0.86],[1]])
    objective_function = Twodimention_problem_2
    D = 2
    bounds_2 = [[0,0], [1,1]] # bounds of Two-dimention problem
    bounds_3 = [[0,0],[5,5]] # bounds of Two-dimention cliff
    bounds_4 = [[-5,0,0],[10,15,1]] # bounds of Three-dimention problem
    bounds_5 = [[-0.95,-0.45], [3.2,4.4]] # bounds of Two-dimention problem Bertsimas
    dists_2 = [Normal(0.085)]*D # distribution of Two-dimention problem
    dists_3 = [Normal(0.5)]*D # distribution of Two-dimention cliff
    dists_4 = [Normal(0.125),Normal(0.125),Normal(0.125)] # distribution of Three-dimention problem
    dists_5 = [Uniform(0.3)]*D
    # dists_4 = [TruncatedNormal(0.125,-5,10),TruncatedNormal(0.125,0,15),TruncatedNormal(0.125,0,1)] # distribution of Three-dimention problem
    
    n_samples = 5*D+5
    N_TRIALS = 10
    n_iterations = 50

    # 二维的初始数据和分布
   
    bounds = bounds_5
    dists = dists_5
    
    # 随机森林无不确定性
    X_samples_RF_all = []
    y_samples_RF_all = []
    # 随机森林输入不确定性
    X_samples_RF_W_all = []
    y_samples_RF_W_all = []
    # 随机森林双重不确定性
    X_samples_RF_GW_all = []
    y_samples_RF_GW_all = []
    # 高斯过程双重不确定性
    X_samples_GP_all = []
    y_samples_GP_all = []
    # 高斯过程输入不确定性
    X_samples_GP_W_all = []
    y_samples_GP_W_all = []

    best_observed_preference_all_RF = []
    best_observed_preference_all_GP = []
    best_observed_preference_all_RF_W = []
    best_observed_preference_all_GP_W = []
    best_observed_preference_all_RF_GW = []
    best_observed_preference_all_GP_GW = []

    # 统计计算时间
    time_GP_all = []
    time_GP_W_all = []
    time_RF_W_all = []
    time_RF_GW_all = []


    regret_all_RF_GW = []
    regret_all_RF_W = []
    regret_all_GP = []
    regret_mu_all_RF_GW = []
    regret_mu_all_RF_W = []
    regret_mu_all_GP = []

    # 距离指标
    distance_all_RF_GW = []
    distance_all_RF_W = []
    distance_all_GP = []
    distance_all_GP_W = []
    distance_mu_all_RF_GW = []
    distance_mu_all_RF_W = []
    distance_mu_all_GP = []
    verbose = False
    for trial in range(1, N_TRIALS + 1):
        # 生成初始样本点
        samples = generate_sample_points(bounds, n_samples=n_samples,D=D)
        # arr = pd.read_excel('/home/Jiangmingqi/experiments/myprogramm2/my_programm2/2D_2.xlsx')

        X_samples_RF = samples
        y_samples_RF = []
        # 随机森林输入不确定性
        X_samples_RF_W = samples
        y_samples_RF_W = []
        # 随机森林双重不确定性
        X_samples_RF_GW = samples
        y_samples_RF_GW = []
        # 高斯过程双重不确定性
        X_samples_GP = samples
        y_samples_GP = []
        # 高斯过程输入不确定性
        X_samples_GP_W = samples
        y_samples_GP_W = []
        model_type = 'rf'
        
        X_samples_GP_list = []
        X_samples_RF_list = []
        
        
        y_samples_GP_list = []
        y_samples_RF_list = []
        # 遗憾指标
        regret_RF_GW_list = []
        regret_RF_W_list = []
        regret_GP_list = []
        regret_RF_GW_mu_list = []
        regret_RF_W_mu_list = []
        regret_GP_mu_list = []


        time_GP = []
        time_RF_GW = []
        time_RF_W = []
        time_GP_W = []
        
        grid_num = 51
        x = np.linspace(bounds[0][0], bounds[1][0], grid_num)  # 生成连续数据
        y = np.linspace(bounds[0][1], bounds[1][1], grid_num)  # 生成连续数据
        x_,y_ = transform_to_t([x,y])
        X, Y = np.meshgrid(x, y)    

        Z = transformed_function([X, Y])
  
        # 三维数据生成
        for xi in X_samples_RF_GW:
            # xi_=np.zeros_like(xi)
            # xi_[0] = xi[0]*(bounds[1][0]-bounds[0][0])+bounds[0][0]
            # xi_[1] = xi[1]*(bounds[1][1]-bounds[0][1])+bounds[0][1]
            # xi_ = transform_to_x(xi) # 如果需要转换回原坐标
            y_samples_RF_GW.append( objective_function(xi))  
        y_samples_RF_GW = np.array(y_samples_RF_GW)
        for xi in X_samples_RF_W:
            # xi_=np.zeros_like(xi)
            # xi_[0] = xi[0]*(bounds[1][0]-bounds[0][0])+bounds[0][0]
            # xi_[1] = xi[1]*(bounds[1][1]-bounds[0][1])+bounds[0][1]
            # xi_ = transform_to_x(xi) 
            y_samples_RF_W.append(objective_function(xi))  
        y_samples_RF_W = np.array(y_samples_RF_W)
        for xi in X_samples_GP:
            # xi_=np.zeros_like(xi)
            # xi_[0] = xi[0]*(bounds[1][0]-bounds[0][0])+bounds[0][0]
            # xi_[1] = xi[1]*(bounds[1][1]-bounds[0][1])+bounds[0][1]
            # xi_ = transform_to_x(xi)
            y_samples_GP.append(objective_function(xi))  
        y_samples_GP = np.array(y_samples_GP)
        
        for x in X_samples_GP_W:
            y_samples_GP_W.append(objective_function(x, uncertainty=0))  
        y_samples_GP_W = np.array(y_samples_GP_W)
        
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        
        best_observed_preference_RF_GW = []
        best_observed_value_RF_GW = y_samples_RF_GW.min()
        best_observed_preference_RF_GW.append(best_observed_value_RF_GW)
        
        best_observed_preference_RF_W = []
        best_observed_value_RF_W = y_samples_RF_W.min()
        best_observed_preference_RF_W.append(best_observed_value_RF_W)
        
        best_observed_preference_GP = []
        best_observed_value_GP = y_samples_GP.min()
        best_observed_preference_GP.append(best_observed_value_GP)
        
        best_observed_preference_GP_W = []
        best_observed_value_GP_W = y_samples_GP_W.min()
        best_observed_preference_GP_W.append(best_observed_value_GP_W)

        X_samples_GP_list = []
        X_samples_RF_list = []
        y_samples_GP_list = []
        y_samples_RF_list = []
        
        # 距离指标
        distance_RF_GW_list = []
        distance_RF_W_list = []
        distance_GP_list = []
        distance_GP_W_list = []
        distance_RF_GW_mu_list = []
        distance_RF_W_mu_list = []
        distance_GP_mu_list = []
        
        for i in range(n_iterations):
            # print(f"Trial: {trial}, Iteration: {i}")
            if X_samples_RF_GW.shape[0] == y_samples_RF_GW.shape[0]:
                # 双重不确定性下随机森林EGO算法
                time_start_RF_GW = time.time()
                
                x_next_RF_GW,y_next_RF_GW,model_3_GW,model_RF_GW,model_std_GW,model_std_mu_GW = robust_optimization_GW_3( objective_function, bounds,
                                                                                    n_samples, n_iterations,distributions=dists,
                                                                                    D=D,samples=X_samples_RF_GW,y_samples=y_samples_RF_GW,uncertainty = 'GW',model_type='rf',goal='min',nproc=1)
                time_end_RF_GW = time.time()
                time_sum_RF_GW = time_end_RF_GW - time_start_RF_GW  
                print('The programm that robust optimiztion using random forest GW using %d 秒'%time_sum_RF_GW)
            time_RF_GW.append(time_sum_RF_GW)
            # 输入不确定性下随机森林EGO算法
            if X_samples_RF_W.shape[0] == y_samples_RF_W.shape[0]:
            
                time_start_RF_W = time.time()
                x_next_RF_W,y_next_RF_W,model_3_W,model_RF_W,model_std_W,model_std_mu_W = robust_optimization_GW_3( objective_function, bounds,
                                                                                    n_samples, n_iterations,distributions=dists,
                                                                                    D=D,samples=X_samples_RF_W,y_samples=y_samples_RF_W,uncertainty = 'W',model_type='rf',goal='min',nproc=1)
                time_end_RF_W = time.time()
                time_sum_RF_W = time_end_RF_W - time_start_RF_W  
                print('The programm that robust optimiztion using random forest W using %d 秒'%time_sum_RF_W)
                time_RF_W.append(time_sum_RF_W)
                # X_samples_RF_GW.shape
                # y_samples_RF_GW.shape
                # X_samples_RF_GW = np.vstack([X_samples_RF_GW, x_next_RF_GW])
                # y_samples_RF_GW = np.hstack([y_samples_RF_GW, y_next_RF_GW])
            # # 双重不确定性下高斯过程EGO算法
            # if X_samples_GP.shape[0] == y_samples_GP.shape[0]:
            #     print(f"Trial: {trial}, Iteration: {i}")
            #     time_start_GP = time.time()
            #     x_next_GP,y_next_GP,model_3_GP,model_RF_GP,model_std_GP,model_std_mu_GP = robust_optimization_GW_3( objective_function, bounds,
            #                                                                         n_samples, n_iterations,distributions=dists,
            #                                                                         D=D,samples=X_samples_GP,y_samples=y_samples_GP,uncertainty = 'GW',model_type='gp',goal='min',nproc=1)
                
            #     time_end_GP = time.time()
            #     time_sum_GP = time_end_GP - time_start_GP  # 计算的时间差为程序的执行时间，单位为秒/s
            #     time_GP.append(time_sum_GP)
            #     print('The programm that robust optimiztion using gaussian process using %d 秒'%time_sum_GP)
            # if X_samples_GP_W.shape[0] == y_samples_GP_W.shape[0]:    
            #     # 输入不确定下高斯过程EGO算法
            #     print(f"Trial: {trial}, Iteration: {i}")
            #     time_start_GP_W = time.time()
            #     x_next_GP_W,y_next_GP_W,model_GP_W,model_RF_GP_W,model_std_W,model_std_mu_W= robust_optimization_GW_3(objective_function, bounds,
            #                                                                         n_samples, n_iterations,distributions=dists,
            #                                                                         D=D,samples=X_samples_GP_W,y_samples=y_samples_GP_W,uncertainty='W',model_type='gp',goal='min',nproc=1)
            #     time_end_GP_W = time.time()
            #     time_sum_GP_W = time_end_GP_W - time_start_GP_W  # 计算的时间差为程序的执行时间，单位为秒/s
            #     print('The programm that robust optimiztion using gaussian process with input uncertainty using %d 秒'%time_sum_GP_W)
            #     time_GP_W.append(time_sum_GP_W)
            
            best_observed_preference_RF_GW.append(min(best_observed_preference_RF_GW[-1],y_next_RF_GW))
            best_observed_preference_RF_W.append(min(best_observed_preference_RF_W[-1],y_next_RF_W))
            # best_observed_preference_GP.append(min(best_observed_preference_GP[-1],y_next_GP))
            # best_observed_preference_GP_W.append(min(best_observed_preference_GP_W[-1],y_next_GP_W))

            x_rm_RF_GW = X_samples_RF_GW[np.argmin(y_samples_RF_GW)]
            x_rm_RF_W = X_samples_RF_W[np.argmin(y_samples_RF_W)]
            x_rm_GP = X_samples_GP[np.argmin(y_samples_GP)]
            
            X_samples_RF_GW = np.vstack([X_samples_RF_GW, x_next_RF_GW])
            y_samples_RF_GW = np.hstack([y_samples_RF_GW, y_next_RF_GW])
            X_samples_RF_W = np.vstack([X_samples_RF_W, x_next_RF_W])
            y_samples_RF_W = np.hstack([y_samples_RF_W, y_next_RF_W])
            # X_samples_GP = np.vstack([X_samples_GP, x_next_GP])
            # y_samples_GP = np.hstack([y_samples_GP, y_next_GP])
            # X_samples_GP_W = np.vstack([X_samples_GP_W, x_next_GP_W])
            # y_samples_GP_W = np.hstack([y_samples_GP_W, y_next_GP_W])

            # regret_RF_GW_mu, regret_RF_GW= Regretfunc(dists,x_rm_RF_GW,model_3_GW,model_RF_GW,model_std_GW,model_std_mu_GW,bounds,goal='min',model_type='rf',uncertainty='GW')
            # regret_RF_W_mu, regret_RF_W= Regretfunc(dists,x_rm_RF_GW,model_3_W,model_RF_W,model_std_W,model_std_mu_W,bounds,goal='min',model_type='rf',uncertainty='W')
            # regret_GP_mu, regret_GP= Regretfunc(dists,x_rm_RF_GW,model_3_GP,model_RF_GP,model_std_GP,model_std_mu_GP,bounds,goal='min',model_type='gp',uncertainty='GW')
            # regret_RF_GW_list.append(regret_RF_GW)
            # regret_RF_GW_mu_list.append(regret_RF_GW_mu)
            # regret_RF_W_list.append(regret_RF_W)
            # regret_RF_W_mu_list.append(regret_RF_W_mu)
            # regret_GP_list.append(regret_GP)
            # regret_GP_mu_list.append(regret_GP_mu)
            print(".", end="")
        

            distance_RF_GW= distance(x_next_RF_GW,goal='min',model_type='rf',uncertainty='GW')
            distance_RF_W= distance(x_next_RF_W,goal='min',model_type='rf',uncertainty='W')
            # distance_GP= distance(x_next_GP,goal='min',model_type='gp',uncertainty='GW')
            # distance_GP_W= distance(x_next_GP_W,goal='min',model_type='gp',uncertainty='W')

            distance_RF_GW_list.append(np.array(distance_RF_GW).flatten())           
            distance_RF_W_list.append(np.array(distance_RF_W).flatten())
            # distance_GP_list.append(np.array(distance_GP).flatten()) 
            # distance_GP_W_list.append(np.array(distance_GP_W).flatten())       

        X_samples_RF_GW_ = [transform_to_t(xn) for xn in X_samples_RF_GW]
        X_samples_RF_W_ = [transform_to_t(xn) for xn in X_samples_RF_W]
        X_samples_GP_ = [transform_to_t(xn) for xn in X_samples_GP]
        X_samples_GP_W_ = [transform_to_t(xn) for xn in X_samples_GP_W]
        mu_wG_RF, sigma_wG_RF = parallel_processing(x, y, dists, model_3_GW, model_RF_GW, model_std_GW,model_std_mu_GW, bounds,fun_RF=process_inner_loop,num_workers=mp.cpu_count(),fun_name ='process_inner_loop')
        mu_w_RF, sigma_w_RF = parallel_processing(x, y, dists, model_3_W, model_RF_GW, model_std_GW,model_std_mu_GW, bounds,fun_RF=process_inner_loop,num_workers=mp.cpu_count(),fun_name ='process_inner_loop',uncertainty='W')
        # mu_wG_GP, sigma_wG_GP = parallel_processing_GP(x, y, dists, model_3_GP, bounds,fun_GP=process_inner_loop_GP,num_workers=mp.cpu_count(),fun_name ='process_inner_loop_GP')
        # mu_w_GP, sigma_w_GP = parallel_processing_GP(x, y, dists, model_GP_W, bounds,fun_GP=process_inner_loop_GP,num_workers=mp.cpu_count(),fun_name ='process_inner_loop_GP')
        # xmin_ro = min_index(mu_wG_RF.reshape(grid_num,grid_num)) 
        # print("ymin_ro=",mu_wG_RF.reshape(grid_num,grid_num)[xmin_ro.flatten()[0],xmin_ro.flatten()[1]])
        # print(xmin_ro)
        # xmin_ro = [x[xmin_ro.flatten()[0]],y[xmin_ro.flatten()[1]]]
        

        time_RF_GW_all.append(time_RF_GW)
        time_RF_W_all.append(time_RF_W)
        time_GP_all.append(time_GP)
        time_GP_W_all.append(time_GP_W)
        # 随机森林输入不确定性
        X_samples_RF_W_all.append(X_samples_RF_W_)
        y_samples_RF_W_all.append(y_samples_RF_W)
        # 随机森林双重不确定性
        X_samples_RF_GW_all.append(X_samples_RF_GW_)
        y_samples_RF_GW_all.append(y_samples_RF_GW)
        # 高斯过程双重不确定性
        X_samples_GP_all.append(X_samples_GP_)
        y_samples_GP_all.append(y_samples_GP)
        # 高斯过程输入不确定性
        X_samples_GP_W_all.append(X_samples_GP_W)
        y_samples_GP_W_all.append(y_samples_GP_W)
        # # 遗憾指标
        # regret_all_RF_GW.append(regret_RF_GW_list)
        # regret_all_RF_W.append(regret_RF_W_list)
        # regret_all_GP.append(regret_GP_list)
        # regret_mu_all_RF_GW.append(regret_RF_GW_mu_list)
        # regret_mu_all_RF_W.append(regret_RF_W_mu_list)
        # regret_mu_all_GP.append(regret_GP_mu_list)
        # 距离指标
        distance_all_RF_GW.append(distance_RF_GW_list)
        distance_all_RF_W.append(distance_RF_W_list)
        distance_all_GP.append(distance_GP_list)
        distance_all_GP_W.append(distance_GP_W_list)
        distance_mu_all_RF_GW.append(distance_RF_GW_mu_list)
        distance_mu_all_RF_W.append(distance_RF_W_mu_list)
        distance_mu_all_GP.append(distance_GP_mu_list)
        print(x_next_RF_GW)
        print(distance_RF_GW)
        best_observed_preference_all_RF_GW.append(best_observed_preference_RF_GW)
        best_observed_preference_all_RF_W.append(best_observed_preference_RF_W) 
        best_observed_preference_all_GP.append(best_observed_preference_GP)
        best_observed_preference_all_GP_W.append(best_observed_preference_GP_W)     
        current_time = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"output_{current_time}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # np.savetxt(folder_name+'/mu_wG_RF.txt', mu_wG_RF.reshape(grid_num,grid_num), delimiter=',')
        # np.savetxt(folder_name+'/sigma_wG_RF.txt', sigma_wG_RF.reshape(grid_num,grid_num), delimiter=',')
        # np.savetxt(folder_name+'/X_samples_RF.txt', X_samples_RF, delimiter=',')
        # np.savetxt(folder_name+'/y_samples_RF.txt', y_samples_RF, delimiter=',')
        # df01 = pd.DataFrame(best_observed_preference_RF_GW)
        # df02 = pd.DataFrame(best_observed_preference_RF_W)
        # df03 = pd.DataFrame(best_observed_preference_GP)
        # df020 = pd.DataFrame(best_observed_preference_GP_W)
        df04 = pd.DataFrame(np.squeeze(mu_wG_RF))
        df05 = pd.DataFrame(np.squeeze(sigma_wG_RF))
        df06 = pd.DataFrame(X_samples_RF_GW)
        df07 = pd.DataFrame(y_samples_RF_GW)
        # df08 = pd.DataFrame(np.squeeze(mu_wG_GP))
        # df09 = pd.DataFrame(np.squeeze(sigma_wG_GP))
        df010 = pd.DataFrame(X_samples_GP)
        df011 = pd.DataFrame(y_samples_GP)
        df021 = pd.DataFrame(X_samples_GP_W)
        df022 = pd.DataFrame(y_samples_GP_W)
        df012 = pd.DataFrame(np.squeeze(mu_w_RF))
        df013 = pd.DataFrame(np.squeeze(sigma_w_RF))
        df014 = pd.DataFrame(X_samples_RF_W)
        df015 = pd.DataFrame(y_samples_RF_W)
        df016 = pd.DataFrame(np.squeeze(distance_all_RF_GW))
        df017 = pd.DataFrame(np.squeeze(distance_all_RF_W))
        df018 = pd.DataFrame(np.squeeze(distance_all_GP))
        df019 = pd.DataFrame(np.squeeze(distance_all_GP_W))
        # df023 = pd.DataFrame(time_RF_GW_all)
        # df024 = pd.DataFrame(time_RF_W_all)
        # df025 = pd.DataFrame(time_GP_all)
        # df026 = pd.DataFrame(time_GP_W_all)
        # df01.to_excel(folder_name+'/best_observed_preference_RF_GW.xlsx')
        # df02.to_excel(folder_name+'/best_observed_preference_RF_W.xlsx')
        # df03.to_excel(folder_name+'/best_observed_preference_GP.xlsx')
        # df020.to_excel(folder_name+'/best_observed_preference_GP_W.xlsx')
        df04.to_excel(folder_name+'/mu_wG_RF.xlsx')
        df05.to_excel(folder_name+'/sigma_wG_RF.xlsx')
        df06.to_excel(folder_name+'/X_samples_RF_GW.xlsx')
        df07.to_excel(folder_name+'/y_samples_RF_GW.xlsx')
        # df08.to_excel(folder_name+'/mu_wG_GP.xlsx')
        # df09.to_excel(folder_name+'/sigma_wG_GP.xlsx')
        df010.to_excel(folder_name+'/X_samples_GP.xlsx')
        df011.to_excel(folder_name+'/y_samples_GP.xlsx')
        df021.to_excel(folder_name+'/X_samples_GP_W.xlsx')
        df022.to_excel(folder_name+'/y_samples_GP_W.xlsx')
        df012.to_excel(folder_name+'/mu_w_RF.xlsx')
        df013.to_excel(folder_name+'/sigma_w_RF.xlsx')
        df014.to_excel(folder_name+'/X_samples_RF_W.xlsx')
        df015.to_excel(folder_name+'/y_samples_RF_W.xlsx')
        df016.to_excel(folder_name+'/distance_all_RF_GW.xlsx')
        df017.to_excel(folder_name+'/distance_all_RF_W.xlsx')
        df018.to_excel(folder_name+'/distance_all_GP.xlsx')
        df019.to_excel(folder_name+'/distance_all_GP_W.xlsx')
        # df023.to_excel(folder_name+'/time_RF_GW_all.xlsx')
        # df024.to_excel(folder_name+'/time_RF_W_all.xlsx')
        # df025.to_excel(folder_name+'/time_GP_all.xlsx')
        # df026.to_excel(folder_name+'/time_GP_W_all.xlsx')
        df1 = pd.DataFrame(best_observed_preference_all_RF_GW)
        df2 = pd.DataFrame(best_observed_preference_all_RF_W)
        df3 = pd.DataFrame(best_observed_preference_all_GP)
        df16 = pd.DataFrame(best_observed_preference_all_GP_W)
        df4 = pd.DataFrame(time_RF_GW_all)
        df5 = pd.DataFrame(time_RF_W_all)
        df6 = pd.DataFrame(time_GP_all)
        df17 = pd.DataFrame(time_GP_W_all)
        # df7 = pd.DataFrame(regret_all_RF_GW)
        # df8 = pd.DataFrame(regret_all_RF_W)
        # df9 = pd.DataFrame(regret_all_GP)
        # df10 = pd.DataFrame(regret_mu_all_RF_GW)
        # df11 = pd.DataFrame(regret_mu_all_RF_W)
        # df12 = pd.DataFrame(regret_mu_all_GP)
        df13 = pd.DataFrame(distance_all_RF_GW)
        df14 = pd.DataFrame(distance_all_RF_W)
        df15 = pd.DataFrame(distance_all_GP)
        df18 = pd.DataFrame(distance_all_GP_W)
        df1.to_excel(folder_name+'/best_observed_preference_all_RF_GW.xlsx')
        df2.to_excel(folder_name+'/best_observed_preference_all_RF_W.xlsx')
        df3.to_excel(folder_name+'/best_observed_preference_all_GP.xlsx')
        df16.to_excel(folder_name+'/best_observed_preference_all_GP_W.xlsx')
        df4.to_excel(folder_name+'/time_RF_GW_all.xlsx')
        df5.to_excel(folder_name+'/time_RF_W_all.xlsx')
        df6.to_excel(folder_name+'/time_GP_all.xlsx')
        df17.to_excel(folder_name+'/time_GP_W_all.xlsx')    

    # df1 = pd.DataFrame(best_observed_preference_all_RF_GW)
    # df2 = pd.DataFrame(best_observed_preference_all_RF_W)
    # df3 = pd.DataFrame(best_observed_preference_all_GP)
    # df16 = pd.DataFrame(best_observed_preference_all_GP_W)
    # df4 = pd.DataFrame(time_RF_GW_all)
    # df5 = pd.DataFrame(time_RF_W_all)
    # df6 = pd.DataFrame(time_GP_all)
    # df17 = pd.DataFrame(time_GP_W_all)
    # # df7 = pd.DataFrame(regret_all_RF_GW)
    # # df8 = pd.DataFrame(regret_all_RF_W)
    # # df9 = pd.DataFrame(regret_all_GP)
    # # df10 = pd.DataFrame(regret_mu_all_RF_GW)
    # # df11 = pd.DataFrame(regret_mu_all_RF_W)
    # # df12 = pd.DataFrame(regret_mu_all_GP)
    # df13 = pd.DataFrame(distance_all_RF_GW)
    # df14 = pd.DataFrame(distance_all_RF_W)
    # df15 = pd.DataFrame(distance_all_GP)
    # df18 = pd.DataFrame(distance_all_GP_W)
    # df1.to_excel(folder_name+'/best_observed_preference_all_RF_GW.xlsx')
    # df2.to_excel(folder_name+'/best_observed_preference_all_RF_W.xlsx')
    # df3.to_excel(folder_name+'/best_observed_preference_all_GP.xlsx')
    # df16.to_excel(folder_name+'/best_observed_preference_all_GP_W.xlsx')
    # df4.to_excel(folder_name+'/time_RF_GW_all.xlsx')
    # df5.to_excel(folder_name+'/time_RF_W_all.xlsx')
    # df6.to_excel(folder_name+'/time_GP_all.xlsx')
    # df17.to_excel(folder_name+'/time_GP_W_all.xlsx')
    # # df7.to_excel(folder_name+'/regret_all_RF_GW.xlsx')
    # # df8.to_excel(folder_name+'/regret_all_RF_W.xlsx')
    # # df9.to_excel(folder_name+'/regret_all_GP.xlsx')
    # # df10.to_excel(folder_name+'/regret_mu_all_RF_GW.xlsx')
    # # df11.to_excel(folder_name+'/regret_mu_all_RF_W.xlsx')
    # # df12.to_excel(folder_name+'/regret_mu_all_GP.xlsx')
    # df13.to_excel(folder_name+'/distance_all_RF_GW.xlsx')
    # df14.to_excel(folder_name+'/distance_all_RF_W.xlsx')
    # df15.to_excel(folder_name+'/distance_all_GP.xlsx')
    # df18.to_excel(folder_name+'/distance_all_GP_W.xlsx')    