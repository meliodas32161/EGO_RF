import numpy as np
import matplotlib.pyplot as plt
from plot_ro import plot_RO_K,plot_RO_RF,plot_RO_K_3,plot_RO_RF_3

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
    f1 = 2*x[0]**6 + 12.2*x[0]**5 + 21.2*x[0]**4 - 6.2*x[0] - 6.4*x[0]**3 - 4.7*x[0]**2 
    f2 = x[1]**6 - 11*x[1]**5 + 43.3*x[1]**4 - 10*x[1] - 74.8*x[1]**3 + 56.9*x[1]**2
    f3 = -4.1*x[0]*x[1] - 0.1*x[0]**2*x[1]**2 + 0.4*x[0]*x[1]**2 + 0.4*x[0]**2*x[1]
    
    return f1+f2+f3

# 三维真实函数——two-dimensional Branin function  + one-dimensional test problem
def Threedimention_cliff(x, uncertainty=0):
    """Twodimention_problem function with input uncertainty."""
    f1 = x[1] - 5.1/(4*np.pi)*x[1] + 5/np.pi*x[0] -6 
    f2 = 10*((1 - 1/(8*np.pi)) * np.cos(x[0]) + 1)
    f3 = (6*x[2] - 2)**2 * np.sin(12*x[2] - 4) + 8*x[2]
    
    return f1+f2+f3


#%% RF 2d
mu_wG_RF = np.loadtxt("output_20241128_160847/mu_wG_RF.txt", delimiter=',')
sigma_wG_RF = np.loadtxt("output_20241128_160847/sigma_wG_RF.txt", delimiter=',')
# mu_RF = np.loadtxt("mu_RF.txt", delimiter=',')

bounds_2 = [[0,0], [1,1]]
grid_num = 51
bounds = bounds_2

x = np.linspace(0, 1, grid_num)  # 生成连续数据
y = np.linspace(0, 1, grid_num)  # 生成连续数据
X, Y = np.meshgrid(x, y)   
Z =Twodimention_problem([X,Y])  
def min_index(data): # 寻找最小值的所有索引
    index = []  # 创建列表,存放最小值的索引
    # data = data.A  # 若data是矩阵，需先转为array,因为矩阵ravel后仍是二维的形式，影响if环节
    dim_1 = data.ravel()  # 转为一维数组
    min_n = max(dim_1)  # 最大值max_n
    for i in range(len(dim_1)):
        if dim_1[i] == min_n:  # 遍历寻找最大值，并全部索引值进行操作
            pos = np.unravel_index(i, data.shape, order='C')  # 返回一维索引对应的多维索引，譬如4把转为(1,1),即一维中的第5个元素对应在二维里的位置是2行2列
            index.append(pos)  # 存入index
    return np.array(index)

# print(mu_wG_RF)
# print(sigma_wG_RF)
xmin_ro = min_index(mu_wG_RF.reshape(grid_num,grid_num)) 
print("ymin_ro=",mu_wG_RF[xmin_ro.flatten()[0],xmin_ro.flatten()[1]])
xmin_ro = [x[xmin_ro.flatten()[1]],y[xmin_ro.flatten()[0]]]
print("xmin_ro=",xmin_ro)
# np.savetxt('mu_wG_RF.txt', mu_wG_RF.reshape(grid_num,grid_num), delimiter=',')
# np.savetxt('sigma_wG_RF.txt', sigma_wG_RF.reshape(grid_num,grid_num), delimiter=',')
fig, (ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5.5))
plt.rc('font',family='Times New Roman', size=12)
# y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3, return_std=True)
# _ = plot_RO_RF_3(ax3, X, Y, Z, np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# _ = plot_RO_K_3(ax1, X, Y, Z, [0, 1], [0, 1], cbar=True)
# _ = plot_RO_RF_3(ax4, X, Y, mu_wG_RF.reshape(21,21), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# _ = plot_RO_RF_3(ax1, X, Y, Z, np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
_ = plot_RO_RF_3(ax2, X, Y, mu_wG_RF.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
_ = plot_RO_RF_3(ax3, X, Y, sigma_wG_RF.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
rp = mu_wG_RF - 2*sigma_wG_RF
_ = plot_RO_RF_3(ax4, X, Y, rp.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
xmin_WG = min_index(rp.reshape(grid_num,grid_num))
print("ymin_WG=",rp.reshape(grid_num,grid_num)[xmin_WG.flatten()[0],xmin_WG.flatten()[1]])
xmin_WG = [x[xmin_WG.flatten()[1]],y[xmin_WG.flatten()[0]]]
print("xmin_WG=",xmin_WG)
# _ = ax3.set_title('objective')
# _ = ax4.set_title('Random forest robust objective')
# # _ = ax3.set_xlabel('x0')
# # _ = ax3.set_ylabel('x1')
# _ = ax4.set_xlabel('x0')
# _ = ax4.set_ylabel('x1')  
# _ = ax1.set_title('objective')
_ = ax2.set_title('Random forest robust objective')
_ = ax3.set_title('Dual uncertainty variance')
_ = ax4.set_title('Dual uncertainty robust objective')
# _ = ax1.set_xlabel('a')
# _ = ax2.set_xlabel('b')
# _ = ax3.set_xlabel('c')
# _ = ax4.set_xlabel('d')
# _ = ax1.set_xlabel('x0')
# _ = ax1.set_ylabel('x1')
_ = ax2.set_xlabel('x1')
_ = ax2.set_ylabel('x2')   
_ = ax3.set_xlabel('x1')
_ = ax3.set_ylabel('x2')
_ = ax4.set_xlabel('x1')
_ = ax4.set_ylabel('x2')

# _ = ax1.text(2.5, -1,'(a)', ha='center')
_ = ax2.text(0.5, -0.2,'(a)', ha='center')
_ = ax3.text(0.5, -0.2,'(b)', ha='center')
_ = ax4.text(0.5, -0.2,'(c)', ha='center')
fig.subplots_adjust(left=0.03,right=1,top=0.965,bottom=0.07,
        wspace=0.03,hspace=0.2)



#%% GP 2d
# mu_wG_GP = np.loadtxt("mu_wG_GP.txt", delimiter=',')
# sigma_wG_GP = np.loadtxt("sigma_wG_GP.txt", delimiter=',')
# bounds_2 = [[0,0], [1,1]]
# grid_num = 51
# bounds = bounds_2

# x = np.linspace(0, 1, grid_num)  # 生成连续数据
# y = np.linspace(0, 1, grid_num)  # 生成连续数据
# X, Y = np.meshgrid(x, y)   
# Z =Twodimention_problem([X,Y])  
# def min_index(data): # 寻找最小值的所有索引
#     index = []  # 创建列表,存放最小值的索引
#     # data = data.A  # 若data是矩阵，需先转为array,因为矩阵ravel后仍是二维的形式，影响if环节
#     dim_1 = data.ravel()  # 转为一维数组
#     min_n = max(dim_1)  # 最大值max_n
#     for i in range(len(dim_1)):
#         if dim_1[i] == min_n:  # 遍历寻找最大值，并全部索引值进行操作
#             pos = np.unravel_index(i, data.shape, order='C')  # 返回一维索引对应的多维索引，譬如4把转为(1,1),即一维中的第5个元素对应在二维里的位置是2行2列
#             index.append(pos)  # 存入index
#     return np.array(index)

# # print(mu_wG_RF)
# # print(sigma_wG_RF)
# xmin_ro = min_index(mu_wG_GP.reshape(grid_num,grid_num)) 
# print("ymin_ro=",mu_wG_GP[xmin_ro.flatten()[0],xmin_ro.flatten()[1]])
# xmin_ro = [x[xmin_ro.flatten()[1]],y[xmin_ro.flatten()[0]]]
# print("xmin_ro=",xmin_ro)
# # np.savetxt('mu_wG_GP.txt', mu_wG_GP.reshape(grid_num,grid_num), delimiter=',')
# # np.savetxt('sigma_wG_GP.txt', sigma_wG_GP.reshape(grid_num,grid_num), delimiter=',')
# fig, (ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5.5))
# plt.rc('font',family='Times New Roman', size=16)
# # y_pred_mean_3,y_pred_std_3 = model_RF.predict(x_true_3, return_std=True)
# # _ = plot_RO_RF_3(ax3, X, Y, Z, np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# # _ = plot_RO_K_3(ax1, X, Y, Z, [0, 1], [0, 1], cbar=True)
# # _ = plot_RO_RF_3(ax4, X, Y, mu_wG_RF.reshape(21,21), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# # _ = plot_RO_RF_3(ax1, X, Y, Z, np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# _ = plot_RO_K_3(ax2, X, Y, mu_wG_GP.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# _ = plot_RO_K_3(ax3, X, Y, sigma_wG_GP.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# rp = mu_wG_GP - 3*sigma_wG_GP
# _ = plot_RO_K_3(ax4, X, Y, rp.reshape(grid_num,grid_num), np.array(bounds).T[0], np.array(bounds).T[1], cbar=True)
# xmin_WG = min_index(rp.reshape(grid_num,grid_num))
# print("ymin_WG=",rp.reshape(grid_num,grid_num)[xmin_WG.flatten()[0],xmin_WG.flatten()[1]])
# xmin_WG = [x[xmin_WG.flatten()[1]],y[xmin_WG.flatten()[0]]]
# print("xmin_WG=",xmin_WG)
# # _ = ax3.set_title('objective')
# # _ = ax4.set_title('Random forest robust objective')
# # # _ = ax3.set_xlabel('x0')
# # # _ = ax3.set_ylabel('x1')
# # _ = ax4.set_xlabel('x0')
# # _ = ax4.set_ylabel('x1')  
# # _ = ax1.set_title('objective')
# _ = ax2.set_title('Gaussian process robust objective')
# _ = ax3.set_title('Dual uncertainty variance')
# _ = ax4.set_title('Dual uncertainty robust objective')
# # _ = ax1.set_xlabel('a')
# # _ = ax2.set_xlabel('b')
# # _ = ax3.set_xlabel('c')
# # _ = ax4.set_xlabel('d')
# # _ = ax1.set_xlabel('x0')
# # _ = ax1.set_ylabel('x1')
# _ = ax2.set_xlabel('x1')
# _ = ax2.set_ylabel('x2')   
# _ = ax3.set_xlabel('x1')
# # _ = ax3.set_ylabel('x2')
# _ = ax4.set_xlabel('x1')
# # _ = ax4.set_ylabel('x2')

# # _ = ax1.text(2.5, -1,'(a)', ha='center')
# _ = ax2.text(0.5, -0.2,'(a)',size=16, ha='center')
# _ = ax3.text(0.5, -0.2,'(b)',size=16, ha='center')
# _ = ax4.text(0.5, -0.2,'(c)', size=16,ha='center')
# fig.subplots_adjust(left=0.03,right=1,top=0.965,bottom=0.07,
#         wspace=0.03,hspace=0.2)