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
import os
import time
import pandas as pd
# from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal,DiscreteLaplace,Poisson,BoundedUniform
from plot_ro import plot_RO_K,plot_RO_RF,plot_RO_K_3,plot_RO_RF_3
from integrated1d import convolute_RF,convolute_K,convolute_RF_w
# from integrated_mc import convolute_RF_1D,convolute_K
# from numba import jit

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

# 二维真实函数 Robust expected improvement for Bayesian optimization

def Twodimention_problem_2(x, uncertainty=0):
    f1 = -2*x[0]**6 + 12.2*x[0]**5-21.2*x[0]**4 + 6.4*x[0]**3 + 4.7*x[0]**2 - 6.2*x[0]
    f2 = -1*x[1]**6 + 11*x[1]**5 - 43.3*x[1]**4 + 74.8*x[1]**3 - 56.9*x[1]**2 + 10*x[1]
    f3 = 4.1*x[0]*x[1] + 0.1*x[0]**2*x[1]**2 - 0.4*x[0]*x[1]**2 - 0.4*x[0]**2*x[1]
    return f1 + f2 + f3

# 二维真实函数——cliff
def Twodimention_cliff(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    obj = 0
    for i in range(len(x)):
        obj = obj+10 / (1+0.3*np.exp(6*x[i])) + 0.2*x[i]**2
    # obj = np.array([obj + 10 / (1+0.3*np.exp(6*x[i])) + 0.2*x[i]**2 for i in range(len(x))])
    return obj

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
    mu_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,uncertainy,goal = args
    n = X.shape[0] 
    

    if model_type == 'gp':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            mu,_,std = convolute_K(X,model,bounds,distribution)
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
        if uncertainy == 'GW':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,model_std_mu=model_std_mu,bound=bounds)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainy == 'W':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std = convolute_RF_w(X,dists=distribution,model=model,model_RF=model_RF,bound=bounds)
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
    
    y_opt = y_
    # print(mu,std)
    if goal == 'min':
        improve = y_opt-(mu[mask]+2*std[mask])- xi
    else:
        improve = (mu[mask]-2*std[mask])-y_opt- xi
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

def gaussian_ei_GW(X,*args):
    
    xi = 0.1
    n_restarts=20
    X_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,uncertainy,goal = args
    n = X.shape[0] 
    

    if model_type=='gp':
        if uncertainy == 'GW':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,_,std = convolute_K(X,model,bounds,distribution)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        # 其实是不需要if这些的，为了便于理解加上去了，但是完全可以通过控制uncertainty参数,使用convolute函数计算模型预测值来避免下面的if语句
        if uncertainy == 'W':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,_,std = convolute_K(X,model,bounds,distribution)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainy == None:
            mu_samples,std_samples = model.predict(X_sample,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,std = model.predict(X,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                    
    elif model_type == 'rf':
        if uncertainy == 'GW':
            mu_samples,_,std_samples = convolute_RF(X_sample,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
                                                    model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,model_std_mu=model_std_mu,
                                        bound=bounds,uncertainty=uncertainy)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainy == 'W':
            mu_samples,std_samples,var_sample = model.predict(X_sample,distributions=distribution,return_std=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                mu,std,var = model.predict(X,distributions=distribution,return_std=True)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainy == None:
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

# 定义采集函数取最大的函数

def propose_location(acquisition ,X_sample, Y_sample, bounds,model=None,model_RF=None,model_std=None,model_std_mu=None, 
                     distributions=None,model_type='gp',n_restarts = 10,uncertainty='GW',goal='min'):
    
    dim = X_sample.shape[1]   # X_sample: Sample locations (n x d). 所以dim = 1
    min_val = 1
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
        
        res = minimize(min_obj, x0=x0, bounds=bound,args=arg,method = 'SLSQP')        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x   
    # x0 = np.random.uniform(bounds[0], bounds[1], size=(20,)) 
    # res = minimize(min_obj, x0=x0, bounds=bound,args=arg,method = 'L-BFGS-B')
    # min_x = res.x        
            
    return min_x.reshape(1, -1)

# =============================考虑模型和参数不确定性的代理鲁棒优化=======================
def robust_optimization_GW(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,samples=None,uncertainty='GW',model_type='gp',goal='min'):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
        model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=1)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=1)
        model_RF = RF_std(n_estimators=4,criterion='squared_error')
    if model_type == 'rf':
        model = Golem(goal=goal, ntrees=4,random_state=42, nproc=1)
        model_std = Golem_std(goal=goal, ntrees=4,random_state=42, nproc=1)
        model_std_mu = Golem_std_mu(goal=goal, ntrees=4,random_state=42, nproc=1)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',min_variance=0.1)
    y_3 = []
    if samples.all != None:
        X_3 = samples
        n_samples = X_3.shape[0]
        for x in X_3:
            y_3.append(onedimention_problem(x, uncertainty=0))  
        # y_3 = np.array(y_3)
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
        y_3 = []
        for x in X_3:
            y_3.append(objective_function(x, uncertainty=0))  
        
        # y_3 =np.array(y_3)     
    if X_3.shape[1] != D:
        raise ValueError("The dimansion of X must equal to D")
    
        
    y_3 =np.array(y_3)  
    y_dim = len(y_samples_RF_GW.shape)  # 目标空间的维度
    x_true_3 = np.linspace(bounds[0], bounds[1], 21).reshape(-1, D)
    y_true_3 = [objective_function(x, 0) for x in x_true_3]

    mu_wG = []
    std_w = []
    std_WG = []
     
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)

        # Plot the current GP model
        y_pred_mean_3, y_pred_std_3 = model.predict(x_true_3, return_std=True)
        # mu_wG = []
        # std_w = []
        # std_WG = []

        # kriging 推荐值
        x_next_3 = propose_location(gaussian_ei_GW, X_3, y_3, bounds,model=model, n_restarts = 10,distributions = dists,
                                    uncertainty=uncertainty,goal=goal)
        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        y_next_3 = objective_function(x_next_3)

        print(x_next_3)
        print(y_next_3)

    if model_type == 'rf':
        dists = distributions
        # model_RF.fit(X_3, y_3.ravel())
        # std_G = model_RF.predict(X_3,return_std=True)[1]
        # model.fit(X_3, y_3)
        # model_.fit(X_3,std_G**2)
        model_RF.fit(X_3, y_3.ravel())
        model.fit(X_3, y_3)
        model_std.fit(X_3,y_3)
        model_std_mu.fit(X_3,y_3)

        y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)


        # RandomForest 推荐值
        # x_next_3 = propose_location(gaussian_ei, model,model_RF,model_,X_3, y_3, bounds,distributions = dists,model_type=model_type, n_restarts = 10)
        x_next_3 = propose_location(gaussian_ei_GW, X_3, y_3, bounds,model,model_RF,model_std,model_std_mu,distributions = dists,
                                    model_type=model_type, n_restarts = 10,uncertainty=uncertainty,goal=goal)

        y_next_3 = objective_function(x_next_3)
        
        print(x_next_3)
        print(y_next_3)
        # plt.figure()
        # y_ei = gaussian_ei(x_true_3,model,model_RF,bounds,dists,model_type)
        # plt.plot(x_true_3.flatten(),y_ei)
        # plt.show()
        
        # _,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)
   
        
    return x_next_3,y_next_3,model,model_RF,model_std,model_std_mu#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3

# =============================不进行优化，单纯建模=======================
def robust_model(X,objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,samples=None,model_type='gp'):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
        model_RF = RF_std(n_estimators=4,criterion='squared_error')
    if model_type == 'rf':
        model = Golem(goal='min', ntrees=4,random_state=42, nproc=1)
        model_ = Golem(goal='min', ntrees=4,random_state=42, nproc=1)
        model_RF = RF_std(n_estimators=4,criterion='squared_error',min_variance=0.01)
    y_3 = []
    if samples.all != None:
        X_3 = samples
        n_samples = X_3.shape[0]
        for x in X_3:
            y_3.append(onedimention_problem(x, uncertainty=0))  
        # y_3 = np.array(y_3)
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
        y_3 = []
        for x in X_3:
            y_3.append(objective_function(x, uncertainty=0))  
        
        # y_3 =np.array(y_3)     
    if X_3.shape[1] != D:
        raise ValueError("The dimansion of X must equal to D")
    
        
    y_3 =np.array(y_3)  
    y_dim = len(y_samples_RF.shape)  # 目标空间的维度
    x_true_3 = np.linspace(bounds[0], bounds[1], 21).reshape(-1, D)
    y_true_3 = [objective_function(x, 0) for x in x_true_3]

    mu_wG = []
    std_w = []
    std_WG = []
     
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
    if model_type == 'gp':
        dists = distributions
        model.fit(X_3, y_3)

        # Plot the current GP model
        y_pred_mean_3, y_pred_std_3 = model.predict(x_true_3, return_std=True)
        # mu_wG = []
        # std_w = []
        # std_WG = []

        # kriging 推荐值
        x_next_3 = propose_location(gaussian_ei_GW, model,model_RF,model_,X_3, y_3, bounds, n_restarts = 10,distributions = dists)
        y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
        y_next_3 = objective_function(x_next_3.flatten())

        print(x_next_3)
        print(y_next_3)

    if model_type == 'rf':
        dists = distributions
        model_RF.fit(X_3, y_3.ravel())
        model.fit(X_3, y_3)
        std_G = model_RF.predict(X_3,return_std=True)[1]
        model_.fit(X_3,std_G**2)
        # Plot the current RF model
        # y_pred_mean_3 = model.forest.predict(x_true_3)
        y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)


        # RandomForest 推荐值
       
        mu,_,std = convolute_RF(X,dists=distributions,model=model,model_RF=model_RF,model_=model_,bound=bounds)
        y = objective_function(X)
        
        # print(x_next_3)
        # print(y_next_3)
        plt.figure()
        mu_wG = mu#+2*std
        # y_ei = gaussian_ei(x_true_3,model,model_RF,bounds,dists,model_type)
        plt.plot(X,mu_wG.reshape([X.shape[0],X.shape[1]]))
        plt.plot(X,y)
        plt.show()
        
        # _,y_pred_std_3 = model_RF.predict(x_true_3,return_std=True)
   
        
    return mu,std#,model,model_RF#,mu_wG, std_WG,std_w,y_pred_mean_3, y_pred_std_3
def transform_to_t(x):
    x1, x2 = x
    t1 = (x1 + 0.95) / 4.15
    t2 = (x2 + 0.45) / 4.85
    return t1, t2

#%% # 一维函数的实验
def Regretfunc(distribution,x_rm,model,model_RF,model_std,model_std_mu,bound,goal):
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
    regret_t_mu,_,regret_t_std = convolute_RF(x_t,distribution,model,model_RF,model_std,model_std_mu,bound)
    regret_rm_mu,_,regret_rm_std = convolute_RF(x_rm,distribution,model,model_RF,model_std,model_std_mu,bound)
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
    x_t = np.array([0.12])
    distance = np.abs(x_t-X_samples)
    # if len(X_samples[0]==2):
    #     x_t = np.array([0.198, 0.085])
    #     x = transform_to_t(X_samples)
    #     distance = np.linalg.norm(x, ord=2)
    
    return distance
if __name__ == '__main__': 
    # samples=np.array([[0.0], [0.083], [0.164],[0.22], [0.43],  [0.63], [0.69], [0.7754],[0.813], [0.92],[0.95]])
    # samples=np.array([[0],  [0.22], [0.39],  [0.63],  [0.86],[1]])
    
    objective_function = onedimention_problem
    bounds_1 = [0, 1]  # Example for a single-variable optimization problem
    bounds_2 = [[0,0], [1,1]]
    n_samples = 13
    N_TRIALS = 10
    n_iterations = 20
    #一维的初始数据和分布
    D = 1
    dists = [Normal(0.07)]
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
    
    
    
    
    # # 二维的初始数据和分布
    # D = 2
    # dists = [Normal(0.07)]*D
    # samples = generate_sample_points(bounds_2, n_samples=n_samples,D=D)
    # X_samples_RF = samples
    # y_samples_RF = []
    # X_samples_GP = samples
    # y_samples_GP = []

    
    best_observed_preference_all_RF = []
    best_observed_preference_all_GP = []
    best_observed_preference_all_RF_W = []
    best_observed_preference_all_GP_W = []
    best_observed_preference_all_RF_GW = []
    best_observed_preference_all_GP_GW = []
    
    x_true_3 = np.linspace(bounds_1[0], bounds_1[1], 50).reshape(-1, D)
    y_true_3 = [objective_function(x, 0) for x in x_true_3]
    
    # grid = np.meshgrid(*[np.linspace(bounds_2[0][j], bounds_2[1][j], 21) for j in range(2)])
    # x_true_3 = np.vstack([axis.flatten() for axis in grid]).T
    # y_true_3 = [Twodimention_problem(x, 0) for x in x_true_3]
    
    # x = np.linspace(0, 1, 21)  # 生成连续数据
    # y = np.linspace(0, 1, 21)  # 生成连续数据
    # X, Y = np.meshgrid(x, y)    
    # Z = X**2 + Y**2 

    
    # plt.scatter()
    # plt.show()
    
    # 随机森林鲁棒均值，单方差和双方差
    mu_wG_RF_all = []
    std_w_RF_all = []
    std_WG_RF_all = []
    # 高斯过程鲁棒均值，单方差和双方差
    mu_wG_GP_all = []
    std_w_GP_all = []
    std_WG_GP_all = []
    # 随机森林模型均值和方差
    y_pred_mean_3_RF_all = []
    y_pred_std_3_RF_all = []
    # 高斯过程模型均值和方差
    y_pred_mean_3_RF_all = []
    y_pred_std_3_RF_all = []
    # 随机森林样本点鲁棒预测
    std_w_sample_RF_all = []
    std_WG_sample_RF_all = []
    # 高斯过程样本点鲁棒预测
    std_w_sample_GP_all = []
    std_WG_sample_GP_all = []
    # 随机森林样本点预测
    y_pred_std_sample_RF_all = []
    # 高斯过程样本点预测
    y_pred_std_sample_GP_all = []
    
    
    
    # 统计计算时间
    time_GP_all = []
    time_GP_W_all = []
    time_RF_W_all = []
    time_RF_GW_all = []
    # EI函数数据
    EI_GP_all = []
    EI_GP_GW_all = []
    EI_RF_all = []
    EI_RF_GW_all = []

    # 距离指标
    distance_all_RF_GW = []
    distance_all_RF_W = []
    distance_all_GP = []
    distance_all_GP_W = []
    distance_mu_all_RF_GW = []
    distance_mu_all_RF_W = []
    distance_mu_all_GP = []
    for trial in range(1, N_TRIALS + 1):
        # 生成初始样本点
        samples=np.array([[0.4], [0.65]])
        # 随机森林无不确定性
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
        
        for x in X_samples_RF_GW:
            y_samples_RF_GW.append(objective_function(x, uncertainty=0))  
        y_samples_RF_GW = np.array(y_samples_RF_GW)
        for x in X_samples_RF_W:
            y_samples_RF_W.append(objective_function(x, uncertainty=0))  
        y_samples_RF_W = np.array(y_samples_RF_W)
        for x in X_samples_GP:
            y_samples_GP.append(objective_function(x, uncertainty=0))  
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
        distss = [Normal(0.01),Normal(0.03),Normal(0.05),Normal(0.1)]
        
        # 随机森林鲁棒均值，单方差和双方差
        mu_wG_RF = []
        std_w_RF = []
        std_WG_RF = []
        # 高斯过程鲁棒均值，单方差和双方差
        mu_wG_GP = []
        std_w_GP = []
        std_WG_GP = []
        # 随机森林模型均值和方差
        y_pred_mean_3_RF = []
        y_pred_std_3_RF = []
        # 高斯过程模型均值和方差
        y_pred_mean_3_GP = []
        y_pred_std_3_GP = []
        # 随机森林样本点鲁棒预测
        std_w_sample_RF = []
        std_WG_sample_RF = []
        # 高斯过程样本点鲁棒预测
        std_w_sample_GP = []
        std_WG_sample_GP = []
        # 随机森林样本点预测
        y_pred_std_sample_RF = []
        # 高斯过程样本点预测
        y_pred_std_sample_GP = []
        
        time_GP = []
        time_GP_W = []
        time_RF_GW = []
        time_RF_W = []
        # EI函数数据
        EI_GP = []
        EI_GP_GW = []
        EI_RF = []
        EI_RF_GW = []

        # 距离指标
        distance_RF_GW_list = []
        distance_RF_W_list = []
        distance_GP_list = []
        distance_GP_W_list = []
        distance_RF_GW_mu_list = []
        distance_RF_W_mu_list = []
        distance_GP_mu_list = []
        for i in range(n_iterations):
            # 输入不确定性下高斯过程EGO算法
            print(f"Trial: {trial}, Iteration: {i}")
            time_start_GP_W = time.time()
            x_next_GP_W,y_next_GP_W,model_GP_W,model_RF_GP_W,model_std_W,model_std_mu_W= robust_optimization_GW(objective_function, bounds_1,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=1,samples=X_samples_GP_W,uncertainty='W',model_type='gp')
            time_end_GP_W = time.time()
            time_sum_GP_W = time_end_GP_W - time_start_GP_W  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using gaussian process with input uncertainty using %d 秒'%time_sum_GP_W)
            time_GP_W.append(time_sum_GP_W)

            # 双重不确定性下高斯过程EGO算法
            print(f"Trial: {trial}, Iteration: {i}")
            time_start_GP = time.time()
            x_next_GP,y_next_GP,model_GP,model_RF_GP,model_std,model_std_mu= robust_optimization_GW(objective_function, bounds_1,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=1,samples=X_samples_GP,model_type='gp')
            time_end_GP = time.time()
            time_sum_GP = time_end_GP - time_start_GP  # 计算的时间差为程序的执行时间，单位为秒/s
            print('The programm that robust optimiztion using gaussian process using %d 秒'%time_sum_GP)
            time_GP.append(time_sum_GP)

            
            # 双重不确定性下随机森林EGO算法
            time_start_RF_GW = time.time()
            x_next_RF_GW,y_next_RF_GW,model_3,model_RF,model_std,model_std_mu = robust_optimization_GW(objective_function, bounds_1,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=1,samples=X_samples_RF_GW,model_type='rf',goal='min')
            time_end_RF_GW = time.time()
            time_sum_RF_GW = time_end_RF_GW - time_start_RF_GW  
            print('The programm that robust optimiztion using random forest GW using %d 秒'%time_sum_RF_GW)
            time_RF_GW.append(time_sum_RF_GW)
            
            # 输入不确定性下随机森林EGO算法
            time_start_RF_W = time.time()
            x_next_RF_W,y_next_RF_W,model_3_W,model_RF_W,model_std_W,model_std_mu_W = robust_optimization_GW(objective_function, bounds_1,
                                                                                n_samples, n_iterations,distributions=dists,
                                                                                D=1,samples=X_samples_RF_W,uncertainty='W',model_type='rf',goal='min')
            time_end_RF_W = time.time()
            time_sum_RF_W = time_end_RF_W - time_start_RF_W  
            print('The programm that robust optimiztion using random forest W using %d 秒'%time_sum_RF_W)
            time_RF_W.append(time_sum_RF_W)
            # x_next_RF_W,y_next_RF_W,model_3,model_RF = robust_optimization_W(objective_function, bounds_1,
            #                                                                     n_samples, n_iterations,distributions=dists,
            #                                                                     D=1,samples=X_samples_RF,model_type='rf')
            distance_RF_GW= distance(x_next_RF_GW,goal='min',model_type='rf',uncertainty='GW')
            distance_RF_W= distance(x_next_RF_W,goal='min',model_type='rf',uncertainty='W')
            distance_GP= distance(x_next_GP,goal='min',model_type='gp',uncertainty='GW')
            distance_GP_W= distance(x_next_GP_W,goal='min',model_type='gp',uncertainty='W')

            distance_RF_GW_list.append(distance_RF_GW)
            distance_RF_W_list.append(distance_RF_W)
            distance_GP_list.append(distance_GP)
            distance_GP_W_list.append(distance_GP_W)
            
            

            best_observed_preference_RF_GW.append(min(best_observed_preference_RF_GW[-1],y_next_RF_GW))
            best_observed_preference_RF_W.append(min(best_observed_preference_RF_W[-1],y_next_RF_W))
            best_observed_preference_GP.append(min(best_observed_preference_GP[-1],y_next_GP))
            best_observed_preference_GP_W.append(min(best_observed_preference_GP_W[-1],y_next_GP_W))
            print(".", end="")

            # # 画图 
            # # # if i%3 == 0 or i == n_iterations-1:
            # if i == n_iterations-1:
            #     # 画Kriging的图
            #     mu_wG,std_w,std_WG=convolute_K(x_true_3,model_GP,bounds_1,dists)
            #     # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            #     y_pred_mean_3,y_pred_std_3 = model_GP.predict(x_true_3, return_std=True)
            #     _,std_w_sample,std_WG_sample = convolute_K(X_samples_GP,model_GP,bounds_1,dists)
            #     _,y_pred_std_sample = model_GP.predict(X_samples_GP, return_std=True)
            #     plot_RO_K(model=model_GP,x_true_3=x_true_3, y_true_3=y_true_3,
            #               y_pred_mean_3=y_pred_mean_3,y_pred_std_3=y_pred_std_3,X_samples=X_samples_GP,
            #               y_samples=y_samples_GP,mu_wG=mu_wG,std_w=std_w,std_WG=std_WG,
            #               y_pred_std_sample = y_pred_std_sample,std_w_sample=std_w_sample,std_WG_sample=std_WG_sample,bound=bounds_1,dists=dists)
                
            #     mu_wG_GP.append(mu_wG)
            #     std_w_GP.append(std_w)
            #     std_WG_GP.append(std_WG)
            #     y_pred_mean_3_GP.append(y_pred_mean_3)
            #     y_pred_std_3_GP.append(y_pred_std_3)
            #     std_w_sample_GP.append(std_w_sample)
            #     std_WG_sample_GP.append(std_WG)
            #     y_pred_std_sample_GP.append(y_pred_std_sample)
                
            #     # 画Random forest的图
            #     mu_wG,std_w,std_WG=convolute_RF(x_true_3,dists,model_3,model_RF,model_std,model_std_mu,bounds_1)
            #     _,y_pred_std_sample = model_RF.predict(X_samples_RF, return_std=True)
            #     _,std_w_sample,std_WG_sample = convolute_RF(X_samples_RF,dists, model_3,model_RF,model_std,model_std_mu,bounds_1)
            #     y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3, return_std=True) #单纯模型预测均值和方差
            #     # print(mu_wG,std_w,std_WG,y_pred_std_sample,std_w_sample,std_WG_sample,y_pred_mean_3,y_pred_std_3)
            #     plot_RO_RF(model=model_3,model_RF=model_RF,x_true_3=x_true_3, y_true_3=y_true_3,
            #               y_pred_mean_3=y_pred_mean_3,y_pred_std_3=y_pred_std_3,X_samples=X_samples_RF,
            #               y_samples=y_samples_RF,mu_wG=mu_wG,std_w=std_w,std_WG=std_WG,
            #               y_pred_std_sample = y_pred_std_sample,std_w_sample=std_w_sample,std_WG_sample=std_WG_sample,bounds=bounds_1,dists=dists)
            #     mu_wG_RF.append(mu_wG)
            #     std_w_RF.append(std_w)
            #     std_WG_RF.append(std_WG)
            #     y_pred_mean_3_RF.append(y_pred_mean_3)
            #     y_pred_std_3_RF.append(y_pred_std_3)
            #     std_w_sample_RF.append(std_w_sample)
            #     std_WG_sample_RF.append(std_WG)
            #     y_pred_std_sample_RF.append(y_pred_std_sample)
                
            # 填充数据集
            X_samples_RF_GW = np.vstack([X_samples_RF_GW, x_next_RF_GW])
            y_samples_RF_GW = np.vstack([y_samples_RF_GW, y_next_RF_GW])
            X_samples_RF_W = np.vstack([X_samples_RF_W, x_next_RF_W])
            y_samples_RF_W = np.vstack([y_samples_RF_W, y_next_RF_W])
            X_samples_GP = np.vstack([X_samples_GP, x_next_GP])
            y_samples_GP = np.vstack([y_samples_GP, y_next_GP])
            X_samples_GP_W = np.vstack([X_samples_GP_W, x_next_GP_W])
            y_samples_GP_W = np.vstack([y_samples_GP_W, y_next_GP_W])
            # # 计算采集函数
            # GP_ei    = gaussian_ei_GW(x_true_3,X_samples_GP,model_GP,model_RF,model_std,model_std_mu,bounds_1,dists,'gp',None,'min')
            # GP_ei_GW = gaussian_ei_GW(x_true_3,X_samples_GP,model_GP,model_RF,model_std,model_std_mu,bounds_1,dists,'gp','GW','min')
            
            # RF_ei    = gaussian_ei_GW(x_true_3,X_samples_RF,model_3,model_RF,model_std,model_std_mu,bounds_1,dists,'rf',None,'min')
            # RF_ei_GW = gaussian_ei_GW(x_true_3,X_samples_RF,model_3,model_RF,model_std,model_std_mu,bounds_1,dists,'rf','GW','min')
            
            # EI_GP.append(GP_ei)
            # EI_GP_GW.append(GP_ei_GW)
            # EI_RF.append(RF_ei)
            # EI_RF_GW.append(RF_ei_GW)
        # 随机森林输入不确定性
        X_samples_RF_W_all.append(X_samples_RF_W)
        y_samples_RF_W_all.append(y_samples_RF_W)
        # 随机森林双重不确定性
        X_samples_RF_GW_all.append(X_samples_RF_GW)
        y_samples_RF_GW_all.append(y_samples_RF_GW)
        # 高斯过程双重不确定性
        X_samples_GP_all.append(X_samples_GP)
        y_samples_GP_all.append(y_samples_GP)
        # 高斯过程输入不确定性
        X_samples_GP_W_all.append(X_samples_GP_W)
        y_samples_GP_W_all.append(y_samples_GP_W)
        
        time_RF_GW_all.append(time_RF_GW)
        time_RF_W_all.append(time_RF_W)
        time_GP_all.append(time_GP)
        time_GP_W_all.append(time_GP_W)
        EI_GP_all.append(EI_GP)
        EI_GP_GW_all.append(EI_GP_GW)
        EI_RF_all.append(EI_RF)
        EI_RF_GW_all.append(EI_RF_GW)
        # def convert_array_to_float64(array):
        #     # 此处我们采用取平均值作为转换策略，但您可以根据需求选择其他策略
        #     if array.size == 1:
        #         return target_type(array[0])
        #     else:
        #         # 对于包含多个值的数组，我们取平均值并转换为目标类型
        #         return target_type(np.mean(array))
        
        best_observed_preference_all_RF_GW.append(best_observed_preference_RF_GW)
        best_observed_preference_all_RF_W.append(best_observed_preference_RF_W) 
        best_observed_preference_all_GP.append(best_observed_preference_GP)
        best_observed_preference_all_GP_W.append(best_observed_preference_GP_W)  

        # 距离指标
        distance_all_RF_GW.append(distance_RF_GW_list)
        distance_all_RF_W.append(distance_RF_W_list)
        distance_all_GP.append(distance_GP_list)
        distance_all_GP_W.append(distance_GP_W_list)

        distance_mu_all_RF_GW.append(distance_RF_GW_mu_list)
        distance_mu_all_RF_W.append(distance_RF_W_mu_list)
        distance_mu_all_GP.append(distance_GP_mu_list)

current_time = time.strftime("%Y%m%d_%H%M%S")
folder_name ="1d" + f"output_{current_time}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
df1 = pd.DataFrame(np.squeeze(distance_all_RF_GW))
df2 = pd.DataFrame(np.squeeze(distance_all_RF_W))
df3 = pd.DataFrame(np.squeeze(distance_all_GP))
df13 = pd.DataFrame(np.squeeze(distance_all_GP_W))
df4 = pd.DataFrame(time_RF_GW_all)
df5 = pd.DataFrame(time_RF_W_all)
df6 = pd.DataFrame(time_GP_all)
df14 = pd.DataFrame(time_GP_W_all)
df7 = pd.DataFrame(best_observed_preference_all_RF_GW)
df8 = pd.DataFrame(best_observed_preference_all_RF_W)
df9 = pd.DataFrame(best_observed_preference_all_GP)
df15 = pd.DataFrame(best_observed_preference_all_GP_W)
df10 = pd.DataFrame(np.squeeze(X_samples_RF_W_all))
df11 = pd.DataFrame(np.squeeze(X_samples_RF_GW_all))
df12 = pd.DataFrame(np.squeeze(X_samples_GP_all))
df16 = pd.DataFrame(np.squeeze(X_samples_GP_W_all))
df1.to_excel(folder_name+'/distance_all_RF_GW.xlsx')
df2.to_excel(folder_name+'/distance_all_RF_W.xlsx')
df3.to_excel(folder_name+'/distance_all_GP.xlsx')
df13.to_excel(folder_name+'/distance_all_GP_W.xlsx')
df4.to_excel(folder_name+'/time_RF_GW_all.xlsx')
df5.to_excel(folder_name+'/time_RF_W_all.xlsx')
df6.to_excel(folder_name+'/time_GP_all.xlsx')
df14.to_excel(folder_name+'/ime_GP_W_all.xlsx')
df7.to_excel(folder_name+'/best_observed_preference_all_RF_GW.xlsx')
df8.to_excel(folder_name+'/best_observed_preference_all_RF_W.xlsx')
df9.to_excel(folder_name+'/best_observed_preference_all_GP.xlsx')
df15.to_excel(folder_name+'/best_observed_preference_all_GP_W.xlsx')
df10.to_excel(folder_name+'/X_samples_RF_W_all.xlsx')
df11.to_excel(folder_name+'/X_samples_RF_GW_all.xlsx')
df12.to_excel(folder_name+'/X_samples_GP_all.xlsx')
df16.to_excel(folder_name+'/X_samples_GP_W_all.xlsx')
            # # 画EI图
            # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # ax1.plot(x_true_3, GP_ei, label='GP_ei')
            # ax1.plot(x_true_3, GP_ei_GW, label='GP_ei_GW')
            # ax1.legend()
            # ax2.plot(x_true_3, RF_ei, label='RF_ei')
            # ax2.plot(x_true_3, RF_ei_GW, label='RF_ei_GW')
            # ax2.legend()

        # 不同的分布对鲁棒代理建模的影响与实际模型进行比较
        # for dist in distss:
        #     dists = [dist]
        #     mu_wG,std_WG = robust_model(x_true_3,objective_function, bounds_1,
        #                                                                         n_samples, n_iterations,distributions=dists,
        #                                                                         D=1,samples=X_samples_RF,model_type='rf')
        # 数据类型转换函数
        
        
        # GP_GW_all = []

        # for i in range(10):
        #     converted_list = []
        #     for item in best_observed_preference_all_GP[i]:
        #         if isinstance(item, np.ndarray):
        #         # 处理numpy.ndarray类型的元素
        #             converted_item = convert_array_to_float64(item)
        #         else:
        #         # 对于非数组元素，我们尝试直接转换（此处假设它们可以转换为float64）
        #         # 注意：如果列表中包含无法转换为float64的类型，此处将引发异常
        #             converted_item = target_type(item)
        #         converted_list.append(converted_item)
        #     GP_GW_all.append(converted_list)
            
        # RF_GW_all = []

        # for i in range(10):
        #     converted_list = []
        #     for item in best_observed_preference_all_RF_GW[i]:
        #         if isinstance(item, np.ndarray):
        #         # 处理numpy.ndarray类型的元素
        #             converted_item = convert_array_to_float64(item)
        #         else:
        #         # 对于非数组元素，我们尝试直接转换（此处假设它们可以转换为float64）
        #         # 注意：如果列表中包含无法转换为float64的类型，此处将引发异常
        #             converted_item = target_type(item)
        #         converted_list.append(converted_item)
        #     RF_GW_all.append(converted_list)
            
        # RF_W_all = []

        # for i in range(10):
        #     converted_list = []
        #     for item in best_observed_preference_all_RF_W[i]:
        #         if isinstance(item, np.ndarray):
        #         # 处理numpy.ndarray类型的元素
        #             converted_item = convert_array_to_float64(item)
        #         else:
        #         # 对于非数组元素，我们尝试直接转换（此处假设它们可以转换为float64）
        #         # 注意：如果列表中包含无法转换为float64的类型，此处将引发异常
        #             converted_item = target_type(item)
        #         converted_list.append(converted_item)
        #     RF_W_all.append(converted_list)
        
        # 样本点all
        # # 随机森林无不确定性
        # X_samples_RF_all.append()
        # y_samples_RF_all = []
        
        
        # # 随机森林鲁棒均值，单方差和双方差
        # mu_wG_RF_all.append(mu_wG_RF)
        # std_w_RF_all.append(std_w_RF)
        # std_WG_RF_all.append(std_WG_RF)
        # # 高斯过程鲁棒均值，单方差和双方差
        # mu_wG_GP_all.append(mu_wG_GP)
        # std_w_GP_all.append(std_w_GP)
        # std_WG_GP_all.append(std_WG_GP)
        # # 随机森林模型均值和方差
        # y_pred_mean_3_RF_all.append(y_pred_mean_3_RF)
        # y_pred_std_3_RF_all.append(y_pred_std_3_RF)
        # # 高斯过程模型均值和方差
        # y_pred_mean_3_RF_all.append(y_pred_mean_3_RF)
        # y_pred_std_3_RF_all.append(y_pred_std_3_RF)
        # # 随机森林样本点鲁棒预测
        # std_w_sample_RF_all.append(std_w_sample_RF)
        # std_WG_sample_RF_all.append(std_WG_sample_RF)
        # # 高斯过程样本点鲁棒预测
        # std_w_sample_GP_all.append(std_w_sample_GP)
        # std_WG_sample_GP_all.append(std_WG_sample_GP)
        # # 随机森林样本点预测
        # y_pred_std_sample_RF_all.append(y_pred_std_sample_RF)
        # # 高斯过程样本点预测
        # y_pred_std_sample_GP_all.append(y_pred_std_sample_GP)

#         import matplotlib.pyplot as plt
# import numpy as np

# # 假设这是你的迭代优化算法函数（这里用模拟数据代替）
# def optimization_algorithm_iteration(iteration_number):
#     # 这里应该是你的算法代码，返回该迭代的目标函数值
#     # 这里我们模拟一个从10递减到1的值，加上一些随机噪声
#     return 10 - iteration_number + np.random.normal(0, 1)

# # 运行一次完整实验的函数（包含所有迭代）
# def run_experiment(num_iterations=10):
#     objective_values = []
#     for i in range(num_iterations):
#         value = optimization_algorithm_iteration(i + 1)
#         objective_values.append(value)
#     return objective_values

# # 设置实验次数和迭代次数
# num_experiments = 10
# num_iterations = 10

# # 存储每次迭代在所有实验中的目标函数值
# all_objective_values = np.zeros((num_experiments, num_iterations))

# # 运行所有实验
# for exp in range(num_experiments):
#     all_objective_values[exp, :] = run_experiment(num_iterations)

# # 计算每次迭代的均值和标准差
# mean_objective_values = np.mean(all_objective_values, axis=0)
# std_objective_values = np.std(all_objective_values, axis=0)

# # 绘制收敛曲线和置信区间图
# plt.figure(figsize=(10, 6))

# # 绘制收敛曲线（均值）
# plt.plot(range(1, num_iterations + 1), mean_objective_values, marker='o', label='Mean Objective Value')

# # 绘制置信区间（假设95%置信区间，即±1.96*标准差）
# confidence_interval = 1.96 * std_objective_values
# plt.fill_between(range(1, num_iterations + 1), mean_objective_values - confidence_interval, mean_objective_values + confidence_interval, alpha=0.3, label='95% Confidence Interval')

# # 设置图表的标签和标题
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.title('Convergence Curve with 95% Confidence Intervals')
# plt.legend()
# plt.grid(True)

# # 显示图表
# plt.show()
#%%
# import pandas
# # 清洗数据
# GP_GW_all = pd.read_excel('GP_GW_all.xlsx')

# RF_GW_all = pd.read_excel('RF_GW_all.xlsx')

# RF_W_all = pd.read_excel('RF_W_all.xlsx')

# GP_GW_all = np.array(GP_GW_all)[:,1:]

# RF_GW_all =np.array(RF_GW_all)[:,1:]

# RF_W_all =np.array(RF_W_all)[:,1:]

# GP_GW = GP_GW_all
# RF_GW = RF_GW_all
# RF_W = RF_W_all

# RF_GW = np.delete(RF_GW_all,7,0)
# RF_GW = np.delete(RF_GW,0,0)
# RF_GW = np.delete(RF_GW,1,0)

# # RF_W = np.delete(RF_W_all,0,0)

# # GP_GW = np.delete(GP_GW_all,9,0)
# # GP_GW = np.delete(GP_GW,4,0)
# # 画收敛曲线图
# from matplotlib.ticker import MultipleLocator
# # plt.figure(figsize=(10, 6),dpi=600,left=0.03)
# fig,ax = plt.subplots(1,1,figsize=(4.5, 6),dpi=600)
# cm = 1/2.54
# # import matplotlib
# legend_font = {"family" : "Times New Roman"}
# plt.rc('font',family='Times New Roman', size=10)
# mean_RF_W = np.mean(RF_W, axis=0)
# std_RF_W = np.std(RF_W, axis=0)

# mean_RF_GW = np.mean(RF_GW, axis=0)
# std_RF_GW = np.std(RF_GW, axis=0)

# mean_GP = np.mean(GP_GW, axis=0)
# std_GP = np.std(GP_GW, axis=0)
# plt.plot(np.arange(0, n_iterations + 1), mean_RF_W, marker='o', label='mean_RF_GW')
# confidence_interval = 1.96 * std_RF_W

# plt.fill_between(np.arange(0, n_iterations + 1), mean_RF_W - confidence_interval, mean_RF_W + confidence_interval, alpha=0.3, label='95% Confidence Interval')

# plt.plot(np.arange(0, n_iterations + 1), mean_RF_GW, marker='v', label='mean_RF_W')
# confidence_interval = 1.96 * std_RF_GW

# plt.fill_between(np.arange(0, n_iterations + 1), mean_RF_GW - confidence_interval, mean_RF_GW + confidence_interval, alpha=0.3, label='95% Confidence Interval')
# plt.plot(np.arange(0, n_iterations + 1), mean_GP, marker='s', label='mean_GP')
# confidence_interval = 1.96 * std_GP

# plt.fill_between(np.arange(0, n_iterations + 1), mean_GP - confidence_interval, mean_GP + confidence_interval, alpha=0.3, label='95% Confidence Interval')
# plt.xlim(0,20)
# plt.legend()
# ax.xaxis.set_major_locator(MultipleLocator(2))
# fig.subplots_adjust(left=0.08,right = 0.985,top = 0.935, bottom = 0.115 )