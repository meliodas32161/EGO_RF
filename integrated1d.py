import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from golem import * 
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import nquad
# import numba as nb
# from numba import jit, types

from scipy import LowLevelCallable
# from scipy import quad_vec
from scipy.special import erf
from scipy.special import j0
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
# 真实函数对应的代理模型响应
def Gaussianprocess_w(W,*args):
    model,x0 = args
    x = W+x0
    x = x.reshape(1,-1)
    result = model.predict(x, return_std=True)
    return result[0]

def Gaussianprocess_G(W,*args):
    model,x0 = args
    x = W+x0
    x = x.reshape(1,-1)
    result = model.predict(x, return_std=True)
    return result[1]

# 要积分的函数
def integrand_mu(w1,*args):
    model,x,dists = args
    w = [w1]
    pw = 1
    for i in range(len(x)):
        pw = pw*dists[i].pdf(w[i],0)
    return Gaussianprocess_w(w,model,x) * pw

def integrand_std(w1,*args):
    model,x,dists = args
    w = [w1]
    pw = 1
    for i in range(len(x)):
        pw = pw*dists[i].pdf(w[i],0)
    return Gaussianprocess_w(w,model,x)**2 * pw

def integrand_std_G(w1,*args):
    model,x,dists = args
    w = [w1]
    pw = 1
    for i in range(len(x)):
        pw = pw*dists[i].pdf(w[i],0)
    return Gaussianprocess_G(w,model,x)**2 * pw

def golem_std_G(X,model,model_RF,model_,dists,bounds):
    """
    该函数是用于计算方差的离散积分计算。将原本的对模型方差的积分以golem算法表示
    X:要计算的对应位置积分的决策变量
    X_train:训练样本
    model:使用golem算法训练的模型
    model_RF:使用skopt训练的模型，用于计算模型方差
    dists:决策变量满足的概率分布
    bounds：决策变量的边界
    """
    # model_ = model   # 为了放置model本身的参数变化，对中间变量model_进行操作
    # model_ = Golem(goal='min', ntrees=4,random_state=42, nproc=1)
    if len(dists) == 1:
        # prediction, std_G = model_RF.predict(X.reshape(-1,1), return_std=True)
        # model_.fit(X_train,std_G**2)
        result_mu_G,result_std_G = model_.predict(X,distributions=dists,return_std=True)
    else:
        # prediction, std_G = model_RF.predict(X, return_std=True)
        # model_.fit(X_train,std_G**2)
        result_mu_G,result_std_G = model_.predict(X,distributions=dists,return_std=True)
    
    return result_mu_G


class Integrand_RF(object):
    def __init__(self,model,model_RF,dists):
        # self.w1 = w1
        # self.w2 = w2
        # model,model_RF,dists = args
        self.model = model
        self.model_RF = model_RF
        self.dists = dists
        # self.w = [w1,w2]
    
    def integrand_std_G_RF(self,x):
        x = np.atleast_2d(x)
        pw = 1
        stds = [dist.std for dist in self.dists]
        w = [np.random.normal(0, j) for j in stds]
        if len(self.dists) == len(x):
            for i, (dist, wi) in enumerate(zip(self.dists, w)):  
                # 注意：这里的 dist.pdf 调用可能需要根据你的分布类进行调整  
                # 假设 dist.pdf 接受一个点作为输入，并返回该点的概率密度  
                pw *= dist.pdf(x[i] + wi,x[i])
        else:  
            raise ValueError("The number of distributions does not match the dimension of x.") 
        prediction, std = self.model_RF.predict(x, return_std=True)
        return std**2 * pw
        # if len(self.dists) == 1:
        #     pw = pw*self.dists[0].pdf(x+w[0],x)
        # else:
        #     for i in range(len(self.dists)): 
        #         pw = pw*self.dists[i].pdf(x[i]+w[i],x[i])
        
        # return self.model_RF.predict(x,return_std=True)[1]**2 * pw    
    

class ProcessClass():
    def __init__(self,dists,model,model_RF,bounds,dim,fun,X_train):
        self.dists = dists
        self.model = model
        self.model_RF = model_RF
        self.bounds = bounds
        self.fun  = fun
        self.dim = dim 
    
    def calculate_nquad(self,x0):
        #result_mu,result_std =  self.model.predict(x0,distributions=self.dists,return_std=True)   
        
        # 考虑模型和参数不确定性的方差
        # bounds = np.array(self.bound).T.tolist()
        if self.dim == 1:
            self.bounds[0][-1] = x0
            # # 防止边界溢出
            # if self.bounds[0][0]<self.bounds[0][-1]: 
                
        else:
            for i in range(self.dim):
                self.bounds[i][-1] = x0[i]
        result_std_G,error_G = nquad(self.fun, self.bounds,opts={"limit":400})


        # return result_mu,result_std,np.sqrt(result_std_G + result_std**2)
        return result_std_G
    
    def calculate_golem(self):
        #result_mu,result_std =  self.model.predict(x0,distributions=self.dists,return_std=True)   
        
        # 考虑模型和参数不确定性的方差
        # bounds = np.array(self.bound).T.tolist()
        model_ = Golem(goal='min', ntrees=4,random_state=42, nproc=1)
        prediction, std_G = model_RF.predict(X, return_std=True)
        model_.fit(X_train,std_G)
        result_mu_G,result_std_G = model_.predict(X,distributions=dists,return_std=True)
        return result_mu_G

class make_compute_results():
    def __init__(self,bounds, model, dists):
        self.bounds = bounds
        self.model = model
        self.dists = dists
    def compute_results(self,x0):
        # 计算mu
        result_mu, error_mu = nquad(integrand_mu, self.bounds, args=(self.model, x0, self.dists))
        mu = result_mu
        
        # 考虑参数不确定性的方差
        result_std, error_std = nquad(integrand_std, self.bounds, args=(self.model, x0, self.dists))
        std = np.sqrt(result_std - mu**2)
        
        # 考虑模型和参数不确定性的方差
        result_std_G, error_G = nquad(integrand_std_G, self.bounds, args=(self.model, x0, self.dists))
        std_G = np.sqrt(result_std_G + result_std - mu**2)
        
        return mu, std, std_G
# 计算高斯过程模型的鲁棒对等问题
def convolute_K(X,model,bound,dists,uncertainty="GW"):
    mu_wG = []
    std_w = []
    std_WG = []
    if len(np.array(bound).shape) == 1:
        bounds = [bound]
    else:
        bounds = np.array(bound).T.tolist()
    if uncertainty == "GW":
        if len(np.shape(X)) == 1:
            # result_mu, error_mu = nquad(integrand_mu, [[-np.inf, np.inf]],args=(model,X))
            result_mu, error_mu = nquad(integrand_mu, bounds,args=(model,X,dists))
            mu_wG.append(result_mu)
            # 考虑参数不确定性的方差
            result_std, error_std = nquad(integrand_std, bounds,args=(model,X,dists)) 
            std_w.append(np.sqrt(result_std - result_mu**2))
            
            # 考虑模型和参数不确定性的方差
            result_std_G, error_G = nquad(integrand_std_G, bounds,args=(model,X,dists))
            std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
        else:
            # 创建带有额外参数的 compute_results 函数，并行x0
            compute_results_with_params = make_compute_results(bounds, model, dists)
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(np.shape(X))) as executor:
                future_to_x0 = {executor.submit(compute_results_with_params.compute_results, x0): x0 for x0 in X}
                for future in concurrent.futures.as_completed(future_to_x0):
                    x0 = future_to_x0[future]
                    try:
                        mu, std, std_G = future.result()
                        mu_wG.append(mu)
                        std_w.append(std)
                        std_WG.append(std_G)
                    except Exception as exc:
                        print(f'{x0} generated an exception: {exc}')
            
    if uncertainty == "W":
        if len(np.shape(X)) == 1:
            # result_mu, error_mu = nquad(integrand_mu, [[-np.inf, np.inf]],args=(model,X))
            result_mu, error_mu = nquad(integrand_mu, bounds,args=(model,X,dists))
            mu_wG.append(result_mu)
            # 考虑参数不确定性的方差
            result_std, error_std = nquad(integrand_std, bounds,args=(model,X,dists)) 
            std_w.append(np.sqrt(result_std - result_mu**2))
            
            # # 考虑模型和参数不确定性的方差
            # result_std_G, error_G = nquad(integrand_std_G, bounds,args=(model,X,dists))
            std_WG.append(np.sqrt(result_std - result_mu**2))
        else:
            for x0 in X:
                result_mu, error_mu = nquad(integrand_mu, bounds,args=(model,x0,dists))
                mu_wG.append(result_mu)
                # 考虑参数不确定性的方差
                result_std, error_std = nquad(integrand_std, bounds,args=(model,x0,dists)) 
                std_w.append(np.sqrt(result_std - result_mu**2))
                
                # # 考虑模型和参数不确定性的方差
                # result_std_G, error_G = nquad(integrand_std_G, bounds,args=(model,x0,dists))
                std_WG.append(np.sqrt(result_std - result_mu**2))
        if uncertainty is None:
            result_mu,result_std = model.predict(X,return_std=True)
            mu_wG.append(result_mu)
            # 如果没有不确定信息，则就是正常的贝叶斯优化，std_w和std_WG等于高斯过程输出的方差
            std_w.append(result_std)
            std_WG.append(result_std)
            
    mu_wG = np.array(mu_wG)
    std_w = np.array(std_w)
    std_WG = np.array(std_WG)
    
    return mu_wG,std_w,std_WG

# 计算随机森林模型的鲁棒对等问题
# @jit(types.Tuple(types.int32, types.Object)(types.Object))
def convolute_RF(X,dists,model,model_RF,model_std,model_std_mu,bound,return_std=True,uncertainty="GW"):
    mu_wG = []
    std_w = []
    std_WG = []
    if len(np.array(bound).shape) == 1:
        bounds = [bound]
    else:
        bounds = np.array(bound).T.tolist()
    dim = len(bounds)
    if uncertainty == "GW":
        if len(np.shape(X)) == 1:
            # 考虑参数不确定性的均值和方差 
            result_mu,result_std,result_var = model.predict(X,distributions=dists,return_std=True)
            result_mu_G_,result_std_G_ = model_RF.predict(X.reshape(1, -1),return_std=True)
            
            mu_wG.append(result_mu)
            std_w.append(result_std)
            
            result_std_G_1 = model_std.predict(X,dists,return_std=False)
            # result_std_G_2 = model_std_mu.predict(X,dists,return_std=False)
            result_std_G_2 = result_var
            result_std_G = result_std_G_1-result_std_G_2
            std_WG.append(np.sqrt(np.exp(result_std**2+result_std_G)))
        else:
            result_mu,result_std,result_var = model.predict(X,dists,return_std=True)
            result_mu_G_,result_std_G_ = model_RF.predict(X,return_std=True)
            result_std_G_1 = model_std.predict(X,dists,return_std=False)
            result_std_G_2 = result_var  
            result_std_G = np.exp(result_std**2+result_std_G_1-result_std_G_2)
            result_std_G2 = result_std_G.copy()
            mu_wG.append(result_mu)
            std_w.append(result_std)
            std_WG.append(np.sqrt(result_std_G2))
    if uncertainty == "W":
        result_mu,result_std,result_var = model.predict(X,dists,return_std=True)
        mu_wG.append(result_mu)
        std_w.append(result_std)
    if uncertainty == None:
        result_mu_G_,result_std_G_ = model_RF.predict(X.reshape(1, -1),return_std=True)
        mu_wG.append(result_mu_G_)
        std_w.append(result_std_G_)
        std_WG.append(result_std_G_)
    mu_wG = np.array(mu_wG)
    std_w = np.array(std_w)
    std_WG = np.array(std_WG)
    return mu_wG,std_w,std_WG

    if len(np.shape(X)) == 1:
        # 考虑参数不确定性的均值和方差 
        result_mu,result_std = model.predict(X,distributions=dists,return_std=True)        
        mu_wG.append(result_mu)
        std_w.append(result_mu)
    else:
    
        for x0 in X:
            # 考虑参数不确定性的均值和方差
            result_mu,result_std = model.predict(x0,dists,return_std=True)
            mu_wG.append(result_mu)
            std_w.append(result_std)

    mu_wG = np.array(mu_wG)
    std_w = np.array(std_w)
    # std_WG = np.array(std_WG)
    return mu_wG,std_w

