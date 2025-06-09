import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from skopt.learning import RandomForestRegressor as RF_std
from golem import * 
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import nquad
from mpl_toolkits.mplot3d import Axes3D
from extensions import BaseDist, Delta, Normal, TruncatedNormal, FoldedNormal

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

def _return_std(X, trees, predictions, min_variance):
    """
    Returns `std(Y | X)`.

    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is `1 / len(trees)`.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    predictions : array-like, shape=(n_samples,)
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.

    Returns
    -------
    std : array-like, shape=(n_samples,)
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.

    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
    std = np.zeros(len(X))

    for tree in trees:
        #tree.tree_ The underlying Tree object. Please refer to help(sklearn.tree._tree.Tree) for attributes of Tree object and Understanding the decision tree structure for basic usage of these attributes.
        var_tree = tree.tree_.impurity[tree.apply(X)]# apply:将forest中的tree应用到 X，返回树叶索引。

        # This rounding off is done in accordance with the
        # adjustment done in section 4.3.3
        # of http://arxiv.org/pdf/1211.0906v2.pdf to account
        # for cases such as leaves with 1 sample in which there
        # is zero variance.
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    return std

# 一维真实函数
def onedimention_problem(x, uncertainty=0):
    """onedimention_problem function with input uncertainty."""
    return (6*x-2)**2 * np.sin(12*x-4)+ 8*x

# 二维真实函数
def Twodimention_problem(x, uncertainty=0):
    """onedimention_problem function with input uncertainty."""
    return 1.9*(1.35+np.exp(x[0])*np.sin(7*x[0])*13*(x[0]-0.6)**2*np.exp(-x[1])*np.sin(7*x[1]))


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



# 定义概率密度函数 p(w_i)
def p_wi(w_i,mean=0,std=0.07):
    # 每个维度的均值为 0，方差为 0.07
    mean_Wi = mean
    std_dev_Wi = std
    return np.exp(-0.5 * ((w_i - mean_Wi) / std_dev_Wi) ** 2) / (std_dev_Wi * np.sqrt(2 * np.pi))

# 定义联合概率密度函数 p(w) = p(w_1) * p(w_2) * ... * p(w_d)  
# 公式（6）
def p_w(w):
    # if D == 1:
    #     return p_wi(w)
    # else:
    #     # return np.prod([p_wi(i) for i in w])
    #     return p_wi(w)
    return p_wi(w)

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

def RandomForest_w(W,*args):
    model,x0 = args
    x = W+x0
    x = x.reshape(1,-1)
    result = model.forest.predict(x)
    return result
def RandomForest_G(W,*args):
    model,x0 = args
    x = W+x0
    x = x.reshape(1,-1)
    mu_ = model.forest.predict(x)
    std_ = _return_std(x,model.forest,mu_,min_variance=1e-6)
    return std_
# 要积分的函数
def integrand_mu(w,*args):
    model,x = args
    # model = args[1]
    # x     = args[2]
    # D = x.shape[0]
    return Gaussianprocess_w(w,model,x) * p_w(w)

def integrand_std(w,*args):
    model,x = args
    # model = args[1]
    # x     = args[2]
    # D = x.shape[0]
    return Gaussianprocess_w(w,model,x)**2 * p_w(w)

def integrand_std_G(w,*args):
    model,x = args
    # model = args[1]
    # x     = args[2]
    # D = x.shape[0]
    return Gaussianprocess_G(w,model,x)**2 * p_w(w)
def integrand_std_G_RF(w,*args):
    model,x = args
    # model = args[1]
    # x     = args[2]
    # D = x.shape[0]
    return RandomForest_G(w,model,x)**2 * p_w(w)

# 计算高斯过程模型的鲁棒对等问题
def convolute_K(X,model):
    mu_wG = []
    std_w = []
    std_WG = []
    if len(np.shape(X)) == 1:
        result_mu, error_mu = nquad(integrand_mu, [[-np.inf, np.inf]],args=(model,X))
        mu_wG.append(result_mu)
        # 考虑参数不确定性的方差
        result_std, error_std = nquad(integrand_std, [[-np.inf, np.inf]],args=(model,X)) 
        std_w.append(np.sqrt(result_std - result_mu**2))
        
        # 考虑模型和参数不确定性的方差
        result_std_G, error_G = nquad(integrand_std_G, [[-np.inf, np.inf]],args=(model,X))
        std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
    else:
    
        for x0 in X:
            result_mu, error_mu = nquad(integrand_mu, [[-np.inf, np.inf]],args=(model,x0))
            mu_wG.append(result_mu)
            # 考虑参数不确定性的方差
            result_std, error_std = nquad(integrand_std, [[-np.inf, np.inf]],args=(model,x0)) 
            std_w.append(np.sqrt(result_std - result_mu**2))
            
            # 考虑模型和参数不确定性的方差
            result_std_G, error_G = nquad(integrand_std_G, [[-np.inf, np.inf]],args=(model,x0))
            std_WG.append(np.sqrt(result_std_G + result_std - result_mu**2))
            
    mu_wG = np.array(mu_wG)
    std_w = np.array(std_w)
    std_WG = np.array(std_WG)
    
    return mu_wG,std_w,std_WG

# 计算随机森林模型的鲁棒对等问题
def convolute_RF(X,dists,model):
    mu_wG = []
    std_w = []
    std_WG = []
    if len(np.shape(X)) == 1:
        # 考虑参数不确定性的均值和方差
        result_mu,result_std = model.predict(X,distributions=dists,return_std=True)
        mu_wG.append(result_mu)
        std_w.append(result_std)
        
        # 考虑模型和参数不确定性的方差
        result_std_G, error_G = nquad(integrand_std_G_RF, [[-np.inf, np.inf]],args=(model,X))
        std_WG.append(np.sqrt(result_std_G + result_std**2))
    else:
    
        for x0 in X:
            # 考虑参数不确定性的均值和方差
            result_mu,result_std = model.predict(X,dists,return_std=True)
            mu_wG.append(result_mu)
            std_w.append(result_std)
            
            # 考虑模型和参数不确定性的方差
            result_std_G, error_G = nquad(integrand_std_G_RF, [[-np.inf, np.inf]],args=(model,x0))
            std_WG.append(np.sqrt(result_std_G + result_std**2))
    mu_wG = np.array(mu_wG)
    std_w = np.array(std_w)
    std_WG = np.array(std_WG)
    return mu_wG,std_w,std_WG

# EI准则
def gaussian_ei(X,*args):
    
    xi = 0.01
    n_restarts=20
    model,bounds,distribution,model_type = args
    n = X.shape[0] 
    

    if model_type == 'gp':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            mu,_,std = convolute_K(X,model)
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
    
            mu,_,std = convolute_RF(X,dists=distribution,model=model)
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
    
    y_opt = np.max(mu)
    
    improve = y_opt - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore
    return values

# 定义采集函数取最大的函数
def propose_location(acquisition ,model, X_sample, Y_sample, bounds,distributions=None,model_type='gp', n_restarts = 10):
    
    dim = X_sample.shape[1]   # X_sample: Sample locations (n x d). 所以dim = 1
    min_val = 1
    min_x = None
    
    def min_obj(X,*args):
        # Minimization objective is the negative acquisition function
        # return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)#.reshape(-1, dim)
        return -acquisition(X,*args)#.reshape(-1, dim)
   
    # Find the best optimum by starting from n_restart different random points.
    arg = (model,bounds,distributions,model_type)
    # 一维
    bound = bounds
    # 二维以上
    # bound = np.array(bounds).T
    for x0 in np.random.uniform(bounds[0], bounds[1], size=(n_restarts, dim)):
        
        res = minimize(min_obj, x0=x0, bounds=[bound],args=arg,method = 'SLSQP')        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    return min_x.reshape(1, -1)

    
# ==============================考虑参数不确定性的代理鲁棒优化=======================
def robust_optimization_W(objective_function,bounds, n_samples, n_iterations,
                          distribution =None, var=0.07, D=3,samples=None):
   
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True) 
    
    model = GaussianProcessRegressor()
    if samples.all != None:
        X_3 = samples
        n_samples = X_3.shape[0]
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
    if X_3.shape[1] != D:
        raise ValueError("The dimansion of X must equal to D")
    y_3 = []
    for x in X_3:
        y_3.append(objective_function(x, uncertainty=0))  
        
    x_true_3 = np.linspace(bounds[0], bounds[1], 100).reshape(-1, D)
    y_true_3 = [objective_function(x, 0) for x in x_true_3]
    

    for i in range(n_iterations):
        model.fit(X_3, y_3)

        # Plot the current GP model
        y_pred_mean_3, y_pred_std_3 = model.predict(x_true_3, return_std=True)
        # 定义积分区间
        lower_limit = -np.inf * np.ones(D)  # 负无穷
        upper_limit = np.inf * np.ones(D)   # 正无穷
        mu_w = []
        std_w = []
        for x0 in x_true_3:
            result_mu, error_mu = nquad(integrand_mu, [[lower_limit, upper_limit]],args=(model,x0))
            mu_w.append(result_mu)
            result_std, error_std = nquad(integrand_std, [[lower_limit, upper_limit]],args=(model,x0))
            std_w.append(np.sqrt(result_std - result_mu**2))
        mu_w = np.array(mu_w)
        std_w = np.array(std_w)
        plt.figure(figsize=(12, 6))
        # Plot the true objective function
        
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        plt.plot(x_true_3, mu_w, 'k-', label=f'g_w Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), mu_w - 2*std_w, mu_w +2*std_w, alpha=0.2)
        plt.legend()
    return mu_w, std_w

# =============================考虑模型和参数不确定性的代理鲁棒优化=======================
def robust_optimization_GW(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,samples=None,model_type='gp'):
   
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
    if model_type == 'rf':
        model = Golem(goal='min', ntrees=100,random_state=42, nproc=1)
        model_RF = RF_std()
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True) 
    model_RF = RF_std()
    # model = GaussianProcessRegressor()
    if samples.all != None:
        X_3 = samples
        n_samples = X_3.shape[0]
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
    if X_3.shape[1] != D:
        raise ValueError("The dimansion of X must equal to D")
    y_3 = []
    for x in X_3:
        y_3.append(objective_function(x, uncertainty=0))  
    # y_3 =np.array(y_3)     
    x_true_3 = np.linspace(bounds[0], bounds[1], 101).reshape(-1, D)
    y_true_3 = [objective_function(x, 0) for x in x_true_3]
    
    
    # bound = np.array(bounds)
    # bound = bound.T
    

    for i in range(n_iterations):
        model.fit(X_3, y_3)

        # Plot the current GP model
        
        # 定义积分区间
        lower_limit = -np.inf * np.ones(D)  # 负无穷
        upper_limit = np.inf * np.ones(D)   # 正无穷
        mu_wG = []
        std_w = []
        std_WG = []
     
        # result_std_G, error_G = nquad(integrand_std, [bounds],args=(model,x0))
        if model_type == 'gp':
            model.fit(X_3, y_3)
    
            # Plot the current GP model
            y_pred_mean_3, y_pred_std_3 = model.predict(x_true_3, return_std=True)
            # 定义积分区间
            lower_limit = -np.inf * np.ones(D)  # 负无穷
            upper_limit = np.inf * np.ones(D)   # 正无穷
            mu_wG = []
            std_w = []
            std_WG = []
            
            # kriging 推荐值
            x_next_3 = propose_location(gaussian_ei, model,X_3, y_3, bounds, n_restarts = 10)
            y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
            y_next_3 = objective_function(x_next_3.flatten())
            X_3 = np.vstack([X_3, x_next_3])
            y_3.append(y_next_3)
            print(x_next_3)
            print(y_next_3)
            
        if model_type == 'rf':
            dists = distributions
            model.fit(X_3, y_3)
            # Plot the current RF model
            y_pred_mean_3 = model.forest.predict(x_true_3)
            y_pred_std_3  = _return_std(x_true_3, model.forest, y_pred_mean_3, min_variance=1e-6)
            
            # RandomForest 推荐值
            x_next_3 = propose_location(gaussian_ei, model,X_3, y_3, bounds,distributions = dists,model_type=model_type, n_restarts = 10)
            # y_next_mean_3 = model.forest.predict(x_next_3)
            y_next_3 = objective_function(x_next_3.flatten())
            X_3 = np.vstack([X_3, x_next_3])
            # y_3 = np.hstack([y_3, y_next_3])
            y_3.append(y_next_3)
            print(x_next_3)
            print(y_next_3)
        
    
    if model_type=='gp':        
        mu_wG,std_w,std_WG=convolute_K(x_true_3,model)
        mu_wG = np.array(mu_wG)
        std_w = np.array(std_w)
        std_WG = np.array(std_WG)    
        # 画图
        plt.figure()
        plt.subplot(2, 2, 1)
        
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        plt.plot(x_true_3, mu_wG, 'k-', label=f'g_w Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), mu_wG - 2*std_w, mu_wG +2*std_w, alpha=0.2)
        plt.legend()
        plt.ylim(-15, 35)
        plt.show()
        
        plt.subplot(2, 2, 2)
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        plt.plot(x_true_3, mu_wG, 'k-', label=f'g_wG Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), mu_wG - 2*std_WG, mu_wG +2*std_WG, alpha=0.2)
        plt.legend()
        
        plt.ylim(-15, 35)
        
        plt.subplot(2, 2, 3)
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        # plt.plot(x_true_3, mu_wG, 'k-', label=f'g_wG Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), y_pred_mean_3.flatten() - 2*y_pred_std_3, y_pred_mean_3.flatten() +2*y_pred_std_3, alpha=0.2)
        plt.legend()
        plt.ylim(-15, 35)
        
        plt.subplot(2, 2, 4)
        plt.plot(x_true_3,y_pred_std_3,label="g_G")
        plt.plot(x_true_3,std_w,label="g_w")
        plt.plot(x_true_3,std_WG,label="g_WG")
        
        _,y_pred_std_3 = model.predict(X_3, return_std=True)
        _,std_w_3,std_WG_3 = convolute_K(X_3, model)
        
        plt.scatter(X_3,y_pred_std_3)
        plt.scatter(X_3,std_w_3)
        plt.scatter(X_3,std_WG_3)
        # plt.scatter(X_3,y_pred_std_3[index])
        # plt.scatter(X_3,std_w[index])
        # plt.scatter(X_3,std_WG[index])
        plt.legend()
        
        # 找到最优设计,最小化问题：
        y_WG = mu_wG +2*std_WG
        xmin_WG = x_true_3[np.argmin(y_WG)]
        y_W = mu_wG +2*std_w
        xmin_W = x_true_3[np.argmin(y_W)]
        print("xmin_WG=",xmin_WG)
        print("ymin_WG=",np.min(y_WG))
        print("xmin_W=",xmin_W)
        print("ymin_W=",np.min(y_W))
    if model_type=='rf':
        mu_wG,std_w,std_WG=convolute_RF(x_true_3,dists,model)
    
        mu_wG = np.array(mu_wG)
        std_w = np.array(std_w)
        std_WG = np.array(std_WG)    
        # 画图
        plt.figure()
        plt.subplot(2, 2, 1)
        
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        plt.plot(x_true_3, mu_wG[0], 'k-', label=f'g_w Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), mu_wG[0] - 2*std_w[0], mu_wG[0] +2*std_w[0], alpha=0.2)
        plt.legend()
        plt.ylim(-15, 35)
        plt.show()
        
        plt.subplot(2, 2, 2)
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        plt.plot(x_true_3, mu_wG[0], 'k-', label=f'g_wG Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), mu_wG[0] - 2*std_WG[0], mu_wG[0] +2*std_WG[0], alpha=0.2)
        plt.legend()
        
        plt.ylim(-15, 35)
        
        plt.subplot(2, 2, 3)
        plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
        # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
        plt.plot(x_true_3, y_pred_mean_3, 'g-', label=f'gaussian Iteration {i + 1}')
        # plt.plot(x_true_3, mu_wG, 'k-', label=f'g_wG Iteration {i + 1}')
        plt.scatter(X_3, y_3, color='r', marker='o', label='Sample')
        plt.fill_between(x_true_3.flatten(), y_pred_mean_3 - 2*y_pred_std_3, y_pred_mean_3 +2*y_pred_std_3, alpha=0.2)
        plt.legend()
        plt.ylim(-15, 35)
        
        plt.subplot(2, 2, 4)
        plt.plot(x_true_3,y_pred_std_3,label="g_G")
        plt.plot(x_true_3,std_w[0],label="g_w")
        plt.plot(x_true_3,std_WG[0],label="g_WG")
        index = []
        j = 0
        # for i in range(101):
        #     if x_true_3[i] == X_3[j]:
        #         index.append(i)
        #         j = j+1
        #     if j == n_iterations+1:
        #         break
        # if model_type == 'gp':   
        #     _,y_pred_std_3 = model.predict(X_3, return_std=True)
        #     _,std_w_3,std_WG_3 = convolute_K(X_3, model)
        # if model_type == 'rf':   
        #     y_pred_3 = model.forest.predict(X_3)
        #     y_pred_std_3 = _return_std(X_3, model.forest, y_pred_3, min_variance=1e-6)
        #     _,std_w_3,std_WG_3 = convolute_RF(X_3,dists, model)
        y_pred_3 = model.forest.predict(X_3)
        y_pred_std_3 = _return_std(X_3, model.forest, y_pred_3, min_variance=1e-6)
        _,std_w_3,std_WG_3 = convolute_RF(X_3,dists, model)
        plt.scatter(X_3,y_pred_std_3)
        plt.scatter(X_3,std_w_3[0])
        plt.scatter(X_3,std_WG_3[0])
        # plt.scatter(X_3,y_pred_std_3[index])
        # plt.scatter(X_3,std_w[index])
        # plt.scatter(X_3,std_WG[index])
        plt.legend()
        
        # 找到最优设计,最小化问题：
        y_WG = mu_wG[0] +2*std_WG[0]
        xmin_WG = x_true_3[np.argmin(y_WG)]
        y_W = mu_wG[0] +2*std_w[0]
        xmin_W = x_true_3[np.argmin(y_W)]
        print("xmin_WG=",xmin_WG)
        print("ymin_WG=",np.min(y_WG))
        print("xmin_W=",xmin_W)
        print("ymin_W=",np.min(y_W))
        
    return mu_wG, std_WG
#===========================多维考虑模型和参数不确定性的代理鲁棒优化=======================
def robust_optimization_GW_2(objective_function,bounds, n_samples, n_iterations,
                          distributions =None, D=3,samples=None,model_type='gp'):
   
    
    kernel = RBF(1.0,(1e-2,1e2))
    if model_type == 'gp':
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True)
    if model_type == 'rf':
        model = Golem(goal='min', ntrees=100,random_state=42, nproc=1)
        model_RF = RF_std()
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.02, normalize_y=True) 
    # model_RF = RF_std()
    # model = GaussianProcessRegressor()
    if samples!= None:
        X_3 = samples
        n_samples = X_3.shape[0]
    else:
        X_3 = generate_sample_points(bounds, n_samples,D=D)
    if X_3.shape[1] != D:
        raise ValueError("The dimansion of X must equal to D")
    y_3 = []
    for x in X_3:
        y_3.append(objective_function(x, uncertainty=0))  
    y_3 =np.array(y_3)    
    # np.linspace(bounds[0], bounds[1], 21).reshape(-1, D)
    # x_true_3 = np.array(np.meshgrid(np.linspace(bounds[0], bounds[1], 21), np.linspace(bounds[0], bounds[1], 21)).T.reshape(-1, D)
    grid = np.meshgrid(*[np.linspace(bounds_2[0][j], bounds_2[1][j], 21) for j in range(2)])
    x_true_3 = np.vstack([axis.flatten() for axis in grid]).T
    y_true_3 = [objective_function(x, 0) for x in x_true_3]
    
    bound = np.array(bounds)
    bound = bound.T
    for i in range(n_iterations):
        if model_type == 'gp':
            model.fit(X_3, y_3)
    
            # Plot the current GP model
            y_pred_mean_3, y_pred_std_3 = model.predict(x_true_3, return_std=True)
            # 定义积分区间
            lower_limit = -np.inf * np.ones(D)  # 负无穷
            upper_limit = np.inf * np.ones(D)   # 正无穷
            # bound_lim = np.array([lower_limit,upper_limit]).T
            # mu_wG = []
            # std_w = []
            # std_WG = []
            
            # kriging 推荐值
            x_next_3 = propose_location(gaussian_ei, model,X_3, y_3, bounds, n_restarts = 10)
            y_next_mean_3, _ = model.predict(x_next_3, return_std=True)
            y_next_3 = np.array(objective_function(x_next_3.flatten()))
            X_3 = np.vstack([X_3, x_next_3])
            y_3 = np.hstack([y_3, y_next_3])
            print(x_next_3)
            print(y_next_3)
            
        if model_type == 'rf':
            dists = distributions
            model.fit(X_3, y_3)
            # Plot the current RF model
            # y_pred_mean_3 = model.forest.predict(x_true_3)
            
            # RandomForest 推荐值
            x_next_3 = propose_location(gaussian_ei, model,X_3, y_3, bounds,distributions = dists,model_type=model_type, n_restarts = 10)
            # y_next_mean_3 = model.forest.predict(x_next_3)
            y_next_3 = objective_function(x_next_3.flatten())
            X_3 = np.vstack([X_3, x_next_3])
            y_3 = np.hstack([y_3, y_next_3])
            print(x_next_3)
            print(y_next_3)
            
        
        # result_std_G, error_G = nquad(integrand_std, bounds,args=(model,x0))
    if model_type == 'gp':
        mu_wG,std_w,std_WG = convolute_RF(x_true_3,model)
    elif model_type == 'rf':
        mu_wG,std_w,std_WG = convolute_RF(x_true_3,distributions,model)
    
    
    # 画图
    fig1=plt.figure(1)
    fig2=plt.figure(2)
    fig3=plt.figure(3)
   
    ax3 = Axes3D(fig1)
    ax3_K = Axes3D(fig2)
    ax3_W = Axes3D(fig3)
  
    xx = np.arange(bound[0][0],bound[0][1]+0.05,0.05)
    yy = np.arange(bound[1][0],bound[1][1]+0.05,0.05)
    X, Y = np.meshgrid(xx, yy)
    Z_true = np.array(y_true_3).reshape(21, 21)
    if model_type == 'gp':
        Z_K,Z_K_std = model.predict(x_true_3,return_std=True)
        Z_K = Z_K.reshape(21, 21)
        Z_K_std = Z_K_std.reshape(21, 21)
        Z_W = mu_wG.reshape(21, 21)
        Z_W_std = std_w.reshape(21, 21)
        Z_WG_std = std_WG.reshape(21, 21)
    if model_type == 'rf':
        Z_K = model.forest.predict(x_true_3)
        Z_K_std = _return_std(x_true_3, model.forest, Z_K, min_variance=1e-6)
        Z_K = Z_K.reshape(21, 21)
        Z_K_std = Z_K_std.reshape(21, 21)
        Z_W = mu_wG[0].reshape(21, 21)
        Z_W_std = std_w[0].reshape(21, 21)
        Z_WG_std = std_WG[0].reshape(21, 21)
    
    
    
    
    #作图
    ax3.plot_surface(X,Y,Z_true,cmap='rainbow')
    ax3_K.plot_surface(X,Y,Z_K,cmap='rainbow')
    ax3_W.plot_surface(X,Y,Z_W,cmap='rainbow')
    
    # 模型+参数
    plt.figure()
    plt.contourf(X,Y,Z_WG_std)
    plt.contour(X,Y,Z_WG_std)
    cset = plt.contourf(X,Y,Z_WG_std,cmap=plt.cm.hot)
    contour = plt.contour(X,Y,Z_WG_std,8,colors='k')
    plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    # 模型
    plt.figure()
    plt.contourf(X,Y,Z_K_std)
    plt.contour(X,Y,Z_K_std)
    cset = plt.contourf(X,Y,Z_K_std,cmap=plt.cm.hot)
    contour = plt.contour(X,Y,Z_K_std,8,colors='k')
    plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    # 参数
    plt.figure()
    plt.contourf(X,Y,Z_W_std)
    plt.contour(X,Y,Z_W_std)
    cset = plt.contourf(X,Y,Z_W_std,cmap=plt.cm.hot)
    contour = plt.contour(X,Y,Z_W_std,8,colors='k')
    plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()
        
        
        
    # 找到最优设计,最小化问题：
    y_WG = mu_wG[0] +2*std_WG[0]
    xmin_WG = x_true_3[np.argmin(y_WG)]
    y_W = mu_wG +2*std_w
    xmin_W = x_true_3[np.argmin(y_W)]
    print("xmin_WG=",xmin_WG)
    print("ymin_WG=",np.min(y_WG))
    print("xmin_W=",xmin_W)
    print("ymin_W=",np.min(y_W))
        
    return mu_wG, std_WG

        
samples=np.array([[0], [0.22], [0.39], [0.63], [0.86],[1]])
bounds_1 = [0, 1]  # Example for a single-variable optimization problem
bounds_2 = [[0,0], [1,1]]
n_samples = 13
n_iterations = 5

# samples = generate_sample_points(bounds_1, n_samples=4 ,D=1)

mu_wG,  std_WG = robust_optimization_GW(onedimention_problem, bounds_1, n_samples, n_iterations,
                                        D=1,samples=samples,model_type='gp') 
dists = [Normal(0.07)]
mu_wG_RF,  std_WG_RF = robust_optimization_GW(onedimention_problem, bounds_1, n_samples, n_iterations,distributions=dists,
                                              D=1,samples=samples,model_type='rf')
n_samples = 50
# mu_wG,  std_WG = robust_optimization_GW_2(Twodimention_problem, bounds_2, n_samples, n_iterations,D=2)   

# dists = []
# for i in range(2):
#     dists.append(Normal(0.07))
# mu_wG,  std_WG = robust_optimization_GW_2(Twodimention_problem, bounds_2, n_samples, n_iterations,distributions=dists,
#                                           D=2,model_type='rf')

