import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from integrated_mc import convolute_RF,convolute_K

def plot_RO_K(model,x_true_3, y_true_3,y_pred_mean_3,y_pred_std_3,X_samples,y_samples,mu_wG,std_w,std_WG,y_pred_std_sample,std_w_sample,std_WG_sample,bound,dists):

     
    # mu_wG,std_w,std_WG=convolute_K(x_true_3,model)
    # mu_wG = np.array(mu_wG)
    # std_w = np.array(std_w)
    # std_WG = np.array(std_WG)    
    # 画图
    cm = 1/2.54
    # legend_font = {"family" : "Times New Roman"}
    plt.rc('font',family='Times New Roman', size=7.5)
    plt.figure(figsize=(16*cm,14*cm),dpi=600)
    plt.subplot(2, 2, 1)
    
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Gaussian process ')
    plt.plot(x_true_3, mu_wG, 'k-', label='Design uncertainty')
    plt.scatter(X_samples,y_samples, color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), mu_wG.flatten() - 1.96*std_w.flatten(), 
                     mu_wG.flatten() +1.96*std_w.flatten(), alpha=0.2)
    plt.xlabel('(a)')
    plt.legend()
    plt.ylim(-10, 35)
    plt.xlim(0, 1)
    # plt.show()
    
    plt.subplot(2, 2, 2)
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Gaussian process ')
    plt.plot(x_true_3, mu_wG, 'k-', label='Dual uncertainty ')
    plt.scatter(X_samples,y_samples, color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), mu_wG.flatten() - 1.96*std_WG.flatten(), 
                     mu_wG.flatten() +1.96*std_WG.flatten(), alpha=0.2)
    plt.xlabel('(b)')
    plt.legend()
    
    plt.ylim(-10, 35)
    plt.xlim(0, 1)
    plt.subplot(2, 2, 3)
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Model uncertainty')
    # plt.plot(x_true_3, mu_wG, 'k-', label=f'g_wG Iteration {i + 1}')
    plt.scatter(X_samples,y_samples,  color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), y_pred_mean_3.flatten() - 1.96*y_pred_std_3, y_pred_mean_3.flatten() +1.96*y_pred_std_3, alpha=0.2)
    plt.xlabel('(c)')
    plt.legend()
    plt.ylim(-10, 35)
    plt.xlim(0, 1)
    
    
    plt.subplot(2, 2, 4)
    # _,y_pred_std = model.predict(X_samples, return_std=True)
    # _,std_w_3,std_WG_3 = convolute_K(X_samples, model,bound,dists)
    
    # plt.plot(x_true_3,y_pred_std_3,label="g_G")
    # plt.plot(x_true_3,std_w,label="g_w")
    # plt.plot(x_true_3,std_WG,label="g_WG")
    plt.plot(x_true_3,y_pred_std_3,label="σ_S")
    plt.plot(x_true_3,std_w,label="σ_δ")
    plt.plot(x_true_3,std_WG,label="σ_δS")
    plt.scatter(X_samples,y_pred_std_sample)
    plt.scatter(X_samples,std_w_sample)
    plt.scatter(X_samples,std_WG_sample)
    # plt.scatter(X_3,y_pred_std_3[index])
    # plt.scatter(X_3,std_w[index])
    # plt.scatter(X_3,std_WG[index])
    plt.xlabel('(d)')
    plt.legend()
    plt.xlim(0, 1)
    # 找到最优设计,最小化问题：
    y_WG = mu_wG +2*std_WG
    xmin_WG = x_true_3[np.argmin(y_WG)]
    y_W = mu_wG +2*std_w
    xmin_W = x_true_3[np.argmin(y_W)]
    print("xmin_WG=",xmin_WG)
    print("ymin_WG=",np.min(y_WG))
    print("xmin_W=",xmin_W)
    print("ymin_W=",np.min(y_W))

def plot_RO_K_3(ax, x1,x2, y, xlims, ylims, vlims=[None, None], alpha=0.5, contour_lines=True, contour_labels=True,
                 labels_fs=12, labels_fmt='%d', n_contour_lines=8, contour_color='k', contour_alpha=1, cbar=False, cmap='RdBu_r'):

     
    # background surface
    if contour_lines is True:
        contours = ax.contour(x1,x2, y, n_contour_lines,
                              colors=contour_color, alpha=contour_alpha) # 画等高线
        if contour_labels is True:
            _ = ax.clabel(contours, inline=True, fontsize=labels_fs, fmt=labels_fmt)
    mappable = ax.imshow(y, extent=[xlims[0],xlims[1],ylims[0],ylims[1]],origin='lower', 
                         cmap=cmap, alpha=alpha, vmin=vlims[0], vmax=vlims[1])# 画热图

    if cbar is True:
        cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5)

    # mark minima
    ax.scatter([x1.flatten()[np.argmax(y)]], [x2.flatten()[np.argmax(y)]],
               s=200, color='white', linewidth=1, edgecolor='k', marker='*', zorder=20)

    ax.set_aspect('equal', 'box')
    return mappable
    
   
def plot_RO_RF(model,model_RF,dists,x_true_3, y_true_3,y_pred_mean_3,y_pred_std_3,X_samples,y_samples,mu_wG,std_w,std_WG,y_pred_std_sample,std_w_sample,std_WG_sample,bounds):
    # mu_wG,std_w,std_WG=convolute_RF(x_true_3,dists,model,model_RF,bounds)
    
    # mu_wG = np.array(mu_wG)
    # std_w = np.array(std_w)
    # std_WG = np.array(std_WG)    
    # 画图
    cm = 1/2.54
    # legend_font = {"family" : "Times New Roman"}
    plt.rc('font',family='Times New Roman', size=7.5)
    plt.figure(figsize=(16*cm,14*cm),dpi=600)
    
    plt.subplot(2, 2, 1)
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Random forest')
    plt.plot(x_true_3, mu_wG, 'k-', label='Design uncertainty')
    plt.scatter(X_samples,y_samples, color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), mu_wG.flatten() - 2*std_w.flatten(), 
                     mu_wG.flatten() +2*std_w.flatten(), alpha=0.2)
    plt.xlabel('(a)')#注意比较和上面面向对象方式的差异
    plt.legend()
    plt.ylim(-10, 35)
    plt.show()
    
    plt.subplot(2, 2, 2)
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Random forest')
    plt.plot(x_true_3, mu_wG, 'k-', label='Dual uncertainty')
    plt.scatter(X_samples,y_samples, color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), mu_wG.flatten() - 2*std_WG.flatten(), 
                     mu_wG.flatten() +2*std_WG.flatten(), alpha=0.2)
    plt.legend()
    plt.xlabel('(b)')
    plt.ylim(-15, 35)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_true_3, y_true_3, 'r-', label='True Function')
    # plt.scatter(x_next_1, y_next_1, color='blue', marker='o', label='next')
    plt.plot(x_true_3, y_pred_mean_3, 'g-', label='Model uncertainty')
    # plt.plot(x_true_3, mu_wG, 'k-', label=f'g_wG Iteration {i + 1}')
    plt.scatter(X_samples,y_samples, color='r', marker='o', label='Sample')
    plt.fill_between(x_true_3.flatten(), y_pred_mean_3 - 2*y_pred_std_3.ravel(), 
                     y_pred_mean_3 +2*y_pred_std_3.ravel(), alpha=0.2)
    plt.xlabel('(c)')
    plt.legend()
    plt.ylim(-15, 35)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_true_3,y_pred_std_3,label="σ_S")
    plt.plot(x_true_3,std_w,label="σ_δ")
    plt.plot(x_true_3,std_WG,label="σ_δS")
    plt.xlabel('(d)')
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
    # y_pred_3,y_pred_std_3 = model_RF.predict(X_samples,return_std=True)
    
    # y_pred_3 = model.forest.predict(X_3)
    # y_pred_std_3 = _return_std(X_3, model.forest, y_pred_3, min_variance=1e-6)
    # _,std_w_3,std_WG_3 = convolute_RF(X_samples,dists, model,model_RF,bounds)
    plt.scatter(X_samples,y_pred_std_sample)
    plt.scatter(X_samples,std_w_sample)
    plt.scatter(X_samples,std_WG_sample)
    # plt.scatter(X_3,y_pred_std_3[index])
    # plt.scatter(X_3,std_w[index])
    # plt.scatter(X_3,std_WG[index])
    plt.legend()
    plt.show()
    
    
    
    # 找到最优设计,最小化问题：
    y_WG = mu_wG +2*std_WG
    xmin_WG = x_true_3[np.argmin(y_WG)]
    y_W = mu_wG +2*std_w
    xmin_W = x_true_3[np.argmin(y_W)]
    print("xmin_WG=",xmin_WG)
    print("ymin_WG=",np.min(y_WG))
    print("xmin_W=",xmin_W)
    print("ymin_W=",np.min(y_W))
    
def plot_RO_RF_3(ax, x1,x2, y, xlims, ylims, vlims=[None, None], alpha=0.5, contour_lines=True, contour_labels=True,
                 labels_fs=12, labels_fmt='%d', n_contour_lines=8, contour_color='k', contour_alpha=1, cbar=False, cmap='RdBu_r'):
    # background surface
    if contour_lines is True:
        contours = ax.contour(x1,x2, y, n_contour_lines,
                              colors=contour_color, alpha=contour_alpha) # 画等高线
        if contour_labels is True:
            _ = ax.clabel(contours, inline=True, fontsize=labels_fs, fmt=labels_fmt)
    mappable = ax.imshow(y, extent=[xlims[0],xlims[1],ylims[0],ylims[1]],origin='lower', 
                         cmap=cmap, alpha=alpha, vmin=vlims[0], vmax=vlims[1])# 画热图

    if cbar is True:
        cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5)

    # mark minima
    ax.scatter([x1.flatten()[np.argmax(y)]], [x2.flatten()[np.argmax(y)]],
               s=200, color='white', linewidth=1, edgecolor='k', marker='*', zorder=20)

    ax.set_aspect('equal', 'box')
    return mappable
    
    
# 高维无法可视化的函数优化过程    
def plot_RO_RF_D(problem,best_observed_preference_all):
    """
    problem:测试问题,获得基准
    best_observed_preference_all：算法解释后所有的结果
    
    """
    # background surface
    if contour_lines is True:
        contours = ax.contour(x1,x2, y, n_contour_lines,
                              colors=contour_color, alpha=contour_alpha) # 画等高线
        if contour_labels is True:
            _ = ax.clabel(contours, inline=True, fontsize=labels_fs, fmt=labels_fmt)
    mappable = ax.imshow(y, extent=[xlims[0],xlims[1],ylims[0],ylims[1]],origin='lower', 
                         cmap=cmap, alpha=alpha, vmin=vlims[0], vmax=vlims[1])# 画热图

    if cbar is True:
        cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5)

    # mark minima
    ax.scatter([x1.flatten()[np.argmin(y)]], [x2.flatten()[np.argmin(y)]],
               s=200, color='white', linewidth=1, edgecolor='k', marker='*', zorder=20)

    ax.set_aspect('equal', 'box')
    return mappable