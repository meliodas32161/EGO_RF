import numpy as np
from scipy.stats import norm
import vegas
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.ensemble import RandomForestRegressor
# from skopt.learning import RandomForestRegressor as RF_std
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
import re

# def jit_integrand_function(integrand_function):
#     jitted_function = nb.njit(integrand_function, nopython=True)

#     #error_model="numpy" -> Don't check for division by zero
#     @cfunc(float64(intc, CPointer(float64)),error_model="numpy",fastmath=True)
#     def wrapped(n, xx):
#         ar = nb.carray(xx, n)
#         return jitted_function(ar[0], ar[1], ar[2])
#     return LowLevelCallable(wrapped.ctypes)
# 真实函数对应的代理模型响应
# def Gaussianprocess_w(W,*args):
#     model,x0 = args
#     x = W+x0
#     x = x.reshape(1,-1)
#     result = model.predict(x, return_std=True)
#     return result[0]

# def Gaussianprocess_G(W,*args):
#     model,x0 = args
#     x = W+x0
#     x = x.reshape(1,-1)
#     result = model.predict(x, return_std=True)
#     return result[1]

# def RandomForest_G(W,*args):
#     model,model_RF,x0 = args
#     x = W+x0
#     x = x.reshape(1,-1)
    
#     mu_,std_ = model_RF.predict(x,return_std=True)
    
#     # std_ = _return_std(X=x,trees=model.forest,predictions=mu_)

#     return std_

# 要积分的函数
# def integrand_mu(w1,w2,w3,*args):
#     model,x = args
#     # model = args[1]
#     # x     = args[2]
#     # D = x.shape[0]
#     w = [w1,w2,w3]
#     pw = 1
#     for i in range(len(x)):
#         pw = pw*Normal(0.125).pdf(w[i],0)
#     return Gaussianprocess_w(w,model,x) * pw

# def integrand_std(w1,w2,w3,*args):
#     model,x = args
#     # model = args[1]
#     # x     = args[2]
#     # D = x.shape[0]
#     w = [w1,w2,w3]
#     pw = 1
#     for i in range(len(x)):
#         pw = pw*Normal(0.125).pdf(w[i],0)
#     return Gaussianprocess_w(w,model,x)**2 * pw

# def integrand_std_G(w1,w2,w3,*args):
#     model,x = args
#     # model = args[1]
#     # x     = args[2]
#     # D = x.shape[0]
#     w = [w1,w2,w3]
#     pw = 1
#     for i in range(len(x)):
#         pw = pw*Normal(0.125).pdf(w[i],0)
#     return Gaussianprocess_G(w,model,x)**2 * pw
# @jit_integrand_function

# def integrand_std_G_RF(w1,w2,w3,*args):
#     model,model_RF,x = args
#     # model = args[-3]
#     # model_RF = args[-2]
#     # x = args[-1]
    
#     # D = x.shape[0]
   
#     w = [w1,w2,w3]
#     pw = 1
#     for i in range(len(x)):
#         pw = pw*Normal(0.125).pdf(w[i],0)
#     def RandomForest_G(W,model,model_RF,x0):
#         # model,model_RF,x0 = args
#         X = W+x0
#         X = X.reshape(1,-1)
        
#         mu_,std_ = model_RF.predict(X,return_std=True)
        
#         # std_ = _return_std(X=x,trees=model.forest,predictions=mu_)

#         return std_
#     return RandomForest_G(w,model,model_RF,x)**2 * pw# Normal(0.125).pdf(w,0)#p_w(w) # 零均值，方差为0.125，在w点处的概率密度



class intergrand_std_K():
    def __init__(self,args):
        self.args = args
        self.X = self.args[0]
        # self.model_RF = args[2]
        self.model = args[1]
        self.dists = args[2]
        
    def Gaussianprocess_w(self,W):
        # model,x0 = args
        x = W+self.X
        x = x.reshape(1,-1)
        result = self.model.predict(x, return_std=True)
        return result[0]

    def Gaussianprocess_G(self,W):
       
        x = W+self.X
        x = x.reshape(1,-1)
        result = self.model.predict(x, return_std=True)
        return result[1]
    
    def integrand_std(self,w):
        # model,x = args
        # model = args[1]
        # x     = args[2]
        # D = x.shape[0]
        # w = [w1,w2,w3]
        pw = 1
        if len(self.X) == 1:
            pw = pw*self.dists[0].pdf(w,0)
        else:
            for i in range(len(self.X)):
                # pw = pw*Normal(0.125).pdf(w[i],0)
                pw = pw*self.dists[i].pdf(w[i],0)
        return self.Gaussianprocess_w(w)**2 * pw
    
    def integrand_mu(self,w):
        # model,x = args
        # model = args[1]
        # x     = args[2]
        # D = x.shape[0]
        # w = [w1,w2,w3]
        pw = 1
        if len(self.X) == 1:
            pw = pw*self.dists[0].pdf(w,0)
        else:
            for i in range(len(self.X)):
                pw = pw*self.dists[i].pdf(w[i],0)
        return self.Gaussianprocess_w(w) * pw
    def integrand_std_G(self,w):
        pw = 1
        if len(self.X) == 1:
            pw = pw*self.dists[0].pdf(w,0)
        else:
            
            for i in range(len(self.X)):
                pw = pw*self.dists[i].pdf(w[i],0)
        return self.Gaussianprocess_G(w)**2 * pw

# 计算高斯过程模型的鲁棒对等问题
def convolute_K(X,model,bound,dists):
    
    
    if len(np.shape(X)) == 1:
        bounds = np.array(bound).T.tolist()
        integ = vegas.Integrator(bounds)
        fun = intergrand_std_K([X,model,dists])
        result_mu = integ(fun.integrand_mu, nitn=15, neval=3000, nproc=1)
        result_mu = str(result_mu)
        result_mu = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_mu)])
        if len(result_mu)!=1:
            result_mu =result_mu.mean()
        mu_wG = result_mu
        # 考虑参数不确定性的方差
        result_std = integ(fun.integrand_std, nitn=15, neval=3000, nproc=1)
        result_std = str(result_std)
        result_std = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_std)])
        if len(result_std)!=1:
            result_std =result_std.mean()
        std_w = np.sqrt(result_std - result_mu**2)
        
        # 考虑模型和参数不确定性的方差
        result_std_G = integ(fun.integrand_std_G, nitn=15, neval=3000, nproc=1)
        
        # std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
        result_std_G = str(result_std_G)
        result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.+\d+', result_std_G)])
        if len(result_std_G)!=1:
            result_std_G =result_std_G.mean()
        std_WG = np.sqrt(result_std_G + result_std - result_mu**2)
    else:
        mu_wG = []
        std_w = []
        std_WG = []
        for x0 in X:
            
            integ = vegas.Integrator(bound)
            fun = intergrand_std_K([x0,model,dists])
            result_mu = integ(fun.integrand_mu, nitn=15, neval=3000, nproc=1)
            result_mu = str(result_mu)
            result_mu = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_mu)])
            # mu_wG = result_mu
            # 考虑参数不确定性的方差
            result_std = integ(fun.integrand_std, nitn=15, neval=3000, nproc=1)
            result_std = str(result_std)
            result_std = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_std)])

            # std_w = np.sqrt(result_std - result_mu**2)
            
            # 考虑模型和参数不确定性的方差
            result_std_G = integ(fun.integrand_std_G, nitn=15, neval=3000, nproc=1)
            
            # std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
            result_std_G = str(result_std_G)
            result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.+\d+', result_std_G)])

            # std_WG = np.sqrt(result_std_G + result_std - result_mu**2)
            mu_wG.append(result_mu)
            std_w.append(np.sqrt(result_std - result_mu**2))
            std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
        mu_wG = np.array(mu_wG)
        std_w = np.array(std_w)
        std_WG = np.array(std_WG)
    
    return mu_wG,std_w,std_WG

# 计算随机森林模型的鲁棒对等问题
# @jit(types.Tuple(types.int32, types.Object)(types.Object))

# def RandomForest_G(W,x_):
#     # model,model_RF,x0 = args
#     x0 = W+x_
#     x0 = x0.reshape(1,-1)
    
#     mu_,std_ = model_RF_.predict(x0,return_std=True)
    
#     # std_ = _return_std(X=x,trees=model.forest,predictions=mu_)

#     return std_
# def integrand_std_G_RF(w,*args):
#     # model,model_RF,x = args
#     # model = args[-3]
#     # model_RF = args[-2]
#     # x = args[-1]
    
#     # D = x.shape[0]
   
#     # w = [w1,w2,w3]
#     pw = 1
#     for i in range(len(x_)):
#         pw = pw*Normal(0.125).pdf(w[i],0)
#     # def RandomForest_G(W):
#     #     # model,model_RF,x0 = args
#     #     x0 = W+X
#     #     x0 = x0.reshape(1,-1)
        
#     #     mu_,std_ = model_RF.predict(x0,return_std=True)
        
#     #     # std_ = _return_std(X=x,trees=model.forest,predictions=mu_)

#     #     return std_
#     return RandomForest_G(w,x_)**2 * pw# Normal(0.125).pdf(w,0)#p_w(w) # 零均值，方差为0.125，在w点处的概率密度

class intergrand_std_RF():
    def __init__(self,args):
        self.args = args
        self.X = self.args[0]
        self.model_RF = args[2]
        self.model = args[1]
        self.dists = args[3]
        
    def RandomForest_G(self,W):
        
        x0 = W+self.X
        x0 = x0.reshape(1,-1)
        
        mu_,std_ = self.model_RF.predict(x0,return_std=True)

        return std_
    
    def integrand_std_G_RF(self,w):
        
        pw = 1
        if len(self.X) == 1:
            pw = pw*self.dists[0].pdf(w,0)
        else:
            
            for i in range(len(self.X)): 
                pw = pw*self.dists[i].pdf(w[i],0)
        
        return self.RandomForest_G(w)**2 * pw
class intergrand_std_RF_algae():
    def __init__(self,args):
        self.args = args
        self.X = self.args[0]    
        self.model = args[1]
        self.model_RF = args[2]
        self.dists = args[3]
        
    def RandomForest_G(self,W):
        
        x0 = W+self.X
        x0 = x0.reshape(1,-1)      
        mu_,std_ = self.model_RF.predict(x0,return_std=True)
        return std_
    
    def integrand_std_G_RF(self,w):
        
        pw = 1
        if len(self.X) == 1:
            pw = pw*self.dists[0].pdf(w,0)
        else:
            
            for i in range(len(self.X)): 
                pw = pw*self.dists[i].pdf(w[i],0)
        
        return self.RandomForest_G(w)**2 * pw   
    
        
        
        
def convolute_RF(X,dists,model,model_RF,bound):
    mu_wG = []
    std_w = []
    std_WG = []
    
    if len(np.shape(X)) == 1:
        # 考虑参数不确定性的均值和方差 result_std_G
        result_mu,result_std = model.predict(X,distributions=dists,return_std=True)   
        
        # 考虑模型和参数不确定性的方差
        bounds = np.array(bound).T.tolist()
        # scipy数值积分
        # result_std_G, error_G = nquad(integrand_std_G_RF, bounds ,args=(model,model_RF,X),opts={"limit":50})
        
        # 用蒙特卡洛方法积分试试
        integ = vegas.Integrator(bounds)
        fun = intergrand_std_RF([X,model,model_RF,dists])
        result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=6)
        result_std_G = str(result_std_G)
        result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.+\d+', result_std_G)])
        # print(result_std_G)
        if len(result_std_G)!=1:
            result_std_G =result_std_G.mean()
        std_WG = np.sqrt(result_std_G + result_std**2)
        mu_wG = result_mu
        std_w = result_std
    else:
    
        for x0 in X:
            # 考虑参数不确定性的均值和方差
            result_mu,result_std = model.predict(x0,distributions=dists,return_std=True)   
            
            # 考虑模型和参数不确定性的方差
            bounds = np.array(bound).T.tolist()
            integ = vegas.Integrator(bounds)
            fun = intergrand_std_RF([x0,model,model_RF,dists])
            result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=6)
            result_std_G = str(result_std_G)
            result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_std_G)])
            # std_WG = np.sqrt(result_std_G + result_std**2)
            mu_wG.append(result_mu)
            std_w.append(result_std)
            std_WG.append(np.sqrt(result_std_G + result_std**2))
            
        mu_wG = np.array(mu_wG)
        std_w = np.array(std_w)
        std_WG = np.array(std_WG)
    
    # std_WG = result_std_G
    
    return mu_wG,std_w,std_WG

def convolute_RF_1D(X,dists,model,model_RF,bound,return_std=True):
    mu_wG = []
    std_w = []
    std_WG = []
    
    if len(np.shape(X)) == 1:
        # 考虑参数不确定性的均值和方差 result_std_G
        result_mu,result_std = model.predict(X,distributions=dists,return_std=True)   
        if return_std:
            # 考虑模型和参数不确定性的方差
            bounds = np.array(bound).T.tolist()
            # scipy数值积分
            # result_std_G, error_G = nquad(integrand_std_G_RF, bounds ,args=(model,model_RF,X),opts={"limit":50})
            
            # 用蒙特卡洛方法积分试试
            integ = vegas.Integrator(bounds)
            fun = intergrand_std_RF([X,model,model_RF,dists])
            result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=6)
            result_std_G = str(result_std_G)
            result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.+\d+', result_std_G)])
            print(result_std_G)
            if len(result_std_G)!=1:
                result_std_G =result_std_G.mean()
            std_WG = np.sqrt(result_std_G + result_std**2)
            mu_wG = result_mu
            std_w = result_std
        else:
            mu_wG = result_mu
            # return mu_wG
        
    else:
        if return_std:
            for x0 in X:
                # 考虑参数不确定性的均值和方差
                result_mu,result_std = model.predict(x0,distributions=dists,return_std=True)   
                
                # 考虑模型和参数不确定性的方差
                bounds = np.array(bound).T.tolist()
                # scipy数值积分
                # result_std_G, error_G = nquad(integrand_std_G_RF, bounds ,args=(model,model_RF,X),opts={"limit":50})
                # monte carlo 积分
                integ = vegas.Integrator(bounds)
                fun = intergrand_std_RF([x0,model,model_RF,dists])
                result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=6)
                result_std_G = str(result_std_G)
                result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_std_G)])
                # std_WG = np.sqrt(result_std_G + result_std**2)
                mu_wG.append(result_mu)
                std_w.append(result_std)
                std_WG.append(np.sqrt(result_std_G + result_std**2))
                
            mu_wG = np.array(mu_wG)
            std_w = np.array(std_w)
            std_WG = np.array(std_WG)
        else:
            for x0 in X:
                # 考虑参数不确定性的均值和方差
                result_mu,result_std = model.predict(x0,distributions=dists,return_std=True)  
                mu_wG.append(result_mu)
            mu_wG = np.array(mu_wG)
            # return mu_wG
    # std_WG = result_std_G
    
    return mu_wG,std_w,std_WG

def convolute_RF_algae(X,dists,model,model_RF,bound):
    mu_wG = []
    std_w = []
    std_WG = []
    
    if len(np.shape(X)) == 1:
        # 考虑参数不确定性的均值和方差 result_std_G
        result_mu,result_std = model.predict(X,distributions=dists,return_std=True)   
        
        # 考虑模型和参数不确定性的方差
        bounds = np.array(bound).T.tolist()
        # scipy数值积分
        # result_std_G, error_G = nquad(integrand_std_G_RF, bounds ,args=(model,model_RF,X),opts={"limit":50})
        
        # 用蒙特卡洛方法积分试试
        integ = vegas.Integrator(bounds)
        fun = intergrand_std_RF_algae([X,model,model_RF,dists])
        result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=8)
        result_std_G = str(result_std_G)
        result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.+\d+', result_std_G)])
        # print(result_std_G)
        if len(result_std_G)!=1:
            result_std_G =result_std_G.mean()
        std_WG = np.sqrt(result_std_G + result_std**2)
        mu_wG = result_mu
        std_w = result_std
    else:
    
        for x0 in X:
            # 考虑参数不确定性的均值和方差
            result_mu,result_std = model.predict(x0,distributions=dists,return_std=True)   
            
            # 考虑模型和参数不确定性的方差
            bounds = np.array(bound).T.tolist()
            integ = vegas.Integrator(bounds)
            fun = intergrand_std_RF_algae([x0,model,model_RF,dists])
            result_std_G = integ(fun.integrand_std_G_RF, nitn=15, neval=3000, nproc=8)
            result_std_G = str(result_std_G)
            result_std_G = np.array([float(match.group()) for match in re.finditer(r'\d+\.\d+', result_std_G)])
            # std_WG = np.sqrt(result_std_G + result_std**2)
            mu_wG.append(result_mu)
            std_w.append(result_std)
            std_WG.append(np.sqrt(result_std_G + result_std**2))
            
        mu_wG = np.array(mu_wG)
        std_w = np.array(std_w)
        std_WG = np.array(std_WG)
    
    # std_WG = result_std_G
    
    return mu_wG,std_w,std_WG
