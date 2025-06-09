import numpy as np
import geatpy as ea
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import warnings
from integrated_mc import convolute_RF,convolute_K,convolute_RF_algae
from scipy.stats import norm
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

# EI准则

def gaussian_ei(X,*args):
    
    xi = 0.02
    n_restarts=20
    model,model_RF,bounds,distribution,model_type,y_ = args
    # n = X.shape[0] 
    

    if model_type == 'gp':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            mu,_,std = convolute_K(X.Phen,model,bounds,dists=distribution)
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
            
            mu,std = model_RF.predict(X.Phen,return_std=True)
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
    y = mu
    values = np.zeros_like(y)
    # values  = np.zeros_like(mu)
    mask = std > 0
    
    y_opt = y_
    # print(mu,std)
    improve = y[mask]-y_opt - xi
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore
    return values


class MyProblem(ea.Problem):
    def __init__(self, model,model_RF,bounds,distribution,model_type,y_,PoolType='Process'):
        # self.X = X
        self.model = model
        self.model_RF = model_RF
        self.bounds = bounds
        self.distribution = distribution
        self.model_type = model_type
        self.y_ = y_
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

        name = 'Myproblem'
        M = 1
        maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        
        Dim = 4
        
        varTypes = [1,0,0,1]#*Dim #[0,0,0,0,0,0]
        
        lb = bounds[0]
        ub = bounds[1]
        lbin = [1]*Dim
        ubin = [1]*Dim
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
        
    # 定义目标函数
    
    def aimFunc(self,Param):
        
        lenx = len(Param.Phen)
        f = gaussian_ei(Param,self.model,self.model_RF,self.bounds,self.distribution,self.model_type,self.y_)
        Param.ObjV = f.reshape(lenx,1) #.reshape(lenx,1)
        
        

        # CV1 = np.array([np.sum(X[:,3:7])-1 ,-np.sum(X[:,3:7])+1])
        # CV2 = np.array([np.sum(X[:,7:10])-1 ,-np.sum(X[:,7:10])+1])
        # CV = np.hstack([np.abs(np.sum(X[:,3:7])-1),np.abs(np.sum(X[:,7:10])-1)])
        # CV = np.hstack([np.abs(X4%5),np.abs(X5%2)])
        #X.CV = CV

        return Param.ObjV#,X.CV






