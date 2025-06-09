import numpy as np
import geatpy as ea
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import warnings
# from integrated_mc import convolute_RF,convolute_K,convolute_RF_algae
from integrated import convolute_RF,convolute_K#,convolute_RF_algae
from scipy.stats import norm
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

# EI准则
def gaussian_ei_GW(X,*args):
    
    xi = 0.1
    n_restarts=20
    X_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,uncertainy,goal = args
    
    if model_type=='gp':
        if uncertainy == 'GW':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainy)     
            mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainy)
            if (mu.ndim != 1):
                mu = mu.flatten()
            # check dimensionality of mu, std so we can divide them below
            if (mu.ndim != 1) or (std.ndim != 1):
                raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                                  "however both must be 1-dimensional. Did you train "
                                  "your model with an (N, 1) vector instead of an "
                                  "(N,) vector?"
                                  .format(mu.ndim, std.ndim))
        if uncertainy == 'W':
            mu_samples,_,std_samples = convolute_K(X_sample,model=model,bound=bounds,dists=distribution,uncertainty=uncertainy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                mu,_,std = convolute_K(X,model,bounds,distribution,uncertainty=uncertainy)
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
                
                mu,_,std = convolute_RF(X,dists=distribution,model=model,model_RF=model_RF,model_std=model_std,
                                        model_std_mu=model_std_mu,bound=bounds,uncertainty=uncertainy)
                if (mu.ndim != 1):
                    mu = mu.flatten()
                if (std.ndim != 1):
                    std = std.flatten()
        elif uncertainy == 'W':
            mu_samples,std_samples,var_samples = model.predict(X_sample,distributions=distribution,return_std=True)
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

def gaussian_lcb_GW(X,*args):
    kappa = 1.96
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

class MyProblem(ea.Problem):
    def __init__(self, X_sample,model,model_RF,model_std,model_std_mu,bounds,distribution,model_type,goal,uncertainty,PoolType='Process'):
        # self.X = X
        self.X_sample=X_sample
        self.model = model
        self.model_RF = model_RF
        self.model_std = model_std
        self.model_std_mu = model_std_mu
        self.bounds = bounds
        self.distribution = distribution
        self.model_type = model_type
        self.uncertainty = uncertainty
        self.goal = goal
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

        name = 'Myproblem'
        M = 1
        if self.goal == 'max':
            maxormins = [-1]
        else:
            maxormins = [1]
         # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）但是EI函数一般都是最大化
        
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
        f = gaussian_lcb_GW(Param.Phen,self.X_sample,self.model,self.model_RF,self.model_std,self.model_std_mu,self.bounds,self.distribution,self.model_type,self.uncertainty,self.goal)
        Param.ObjV = f.reshape(lenx,1) #.reshape(lenx,1)
        
        

        # CV1 = np.array([np.sum(X[:,3:7])-1 ,-np.sum(X[:,3:7])+1])
        # CV2 = np.array([np.sum(X[:,7:10])-1 ,-np.sum(X[:,7:10])+1])
        # CV = np.hstack([np.abs(np.sum(X[:,3:7])-1),np.abs(np.sum(X[:,7:10])-1)])
        # CV = np.hstack([np.abs(X4%5),np.abs(X5%2)])
        #X.CV = CV

        return Param.ObjV#,X.CV






