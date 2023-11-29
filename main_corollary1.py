import pandas as pd
import numpy as np
from generatedata_v2 import *
import time
import csv
from datetime import datetime
# from opt_v11 import *
from opt_v15_alpha_star import *
from tqdm import tqdm

s1 = {'a_b':np.array([-6,-2,2]),'n0':4000, 'n1': 800, 'p':0.037,'gamma_true':3.259}
s2 = {'a_b':np.array([-5,-2,2]),'n0':4000, 'n1': 800, 'p':0.067,'gamma_true':2.634}
s3 = {'a_b':np.array([-4,2,1]),'n0':4000, 'n1': 800, 'p':0.081,'gamma_true':2.429}
s4 = {'a_b':np.array([-4,2,2]),'n0':4000, 'n1': 800, 'p':0.116,'gamma_true':2.031}

s5 = {'a_b':np.array([-6,-2,2]),'n0':3000, 'n1': 1500, 'p':0.037,'gamma_true':3.259}
s6 = {'a_b':np.array([-5,-2,2]),'n0':3000, 'n1': 1500, 'p':0.067,'gamma_true':2.634}
s7 = {'a_b':np.array([-4,2,1]),'n0':3000, 'n1': 1500, 'p':0.081,'gamma_true':2.429}
s8 = {'a_b':np.array([-4,2,2]),'n0':3000, 'n1': 1500, 'p':0.116,'gamma_true':2.031}

s9 = {'a_b':np.array([-6,-2,2]),'n0':2000, 'n1': 2000, 'p':0.037,'gamma_true':3.259}
s10 = {'a_b':np.array([-5,-2,2]),'n0':2000, 'n1': 2000, 'p':0.067,'gamma_true':2.634}
s11 = {'a_b':np.array([-4,2,1]),'n0':2000, 'n1': 2000, 'p':0.081,'gamma_true':2.429}
s12 = {'a_b':np.array([-4,2,2]),'n0':20, 'n1': 20, 'p':0.116,'gamma_true':2.031}

setting = [s12]
# setting = [s2]
summary = pd.DataFrame()

for sets in tqdm(setting):
    n0 = sets['n0']
    n1 = sets['n1']
    a = sets['a_b'][0]
    b1 = sets['a_b'][1]
    b2 = sets['a_b'][2]
    gamma_true = sets['gamma_true']
    p1 = sets['p']
    factor = 1

    gamma = []
    count = []
    count2 = []
    count3 = []
    theta_all = []
    v_all = []
    cp_all = []
    bias_all = []
    ese_all_W = []
    ese_all_Vhat = []
    ese_alpha_W = []
    ese_alpha_Vhat = []
    v1_all = []
    pbias = []
    result = []
    result2 = []

    
    v_initial=np.array([n1/(n0+n1),0,0])
    theta_true = np.array([gamma_true, gamma_true+a,b1,b2,0,0])
    data = GenerateData([a,b1,b2], 2000, n0, n1, factor) # set parameter

    times = 1300
    for j in range(times):
        # np.random.seed(j)
        X,y,Xy,gamma0 = data.generate(2,0)
        gamma.append(gamma0)

        theta_guess = np.array([gamma_true, a+gamma_true,b1,b2,0,0])+np.random.normal(0,0.05) # theta的初始值=gamma,alpha,beta1,beta2,miu1,miu2
    
        v = v_initial
        theta = theta_guess

        W = np.array([[0.2,0],[0,2]])
        obj = part(X,y,W)
        iter,res_theta,res_v = obj.estimate(theta,v)

        count.append(iter)
        result.append(res_theta.success)
        ese_all_W.append(ese_all_W,obj.ese(res_theta.x))
        ese_alpha_W.append(obj.ese_of_alpha(res_theta.x))
        

        iteration = 1
        max_iter = 100
        epsilon = 1e-3

        while True:
            theta0 = res_theta.x
            V_hat = obj.V_hat(res_theta.x,res_v)
            obj = part(X,y,V_hat)
            iter,res_theta,res_v = obj.estimate(theta,v)
            theta1 = res_theta.x
        
            if np.max(np.abs(theta1 - theta0)) < epsilon: #1范数
                break

            iteration += 1
            if iteration > max_iter:
                break

        count2.append(iter)
        result2.append(res_theta.success)
        count3.append(iteration)
        ese_all_Vhat.append(ese_all_Vhat,obj.ese(res_theta.x))
        ese_alpha_Vhat.append(obj.ese_of_alpha(res_theta.x))


    df_count_W = pd.DataFrame(count,columns=['iterations_W'])
    df_count_Vhat = pd.DataFrame(count,columns=['iterations_Vhat'])
    df_count_iner_iteration = pd.DataFrame(count,columns=['iner_iteration'])

    df_result_W = pd.DataFrame(result,columns=['_Result_'])
    df_result_Vhat = pd.DataFrame(result2,columns=['_Result2_'])


    df_ese_W = pd.DataFrame(ese_all_W,columns=['ese_gamma_W','ese_alpha*_W','ese_beta1_W','ese_beta2_W','ese_miu1_W','ese_miu2_W'])
    df_alpha_ese_W = pd.DataFrame({'ese_alpha_W': ese_alpha_W})
    # df_W = pd.concat([df_ese_W,df_alpha_ese_W],axis =1 )

    df_ese_Vhat = pd.DataFrame(ese_all_Vhat,columns=['ese_gamma_Vhat','ese_alpha*_Vhat','ese_beta1_Vhat','ese_beta2_Vhat','ese_miu1_Vhat','ese_miu2_Vhat'])
    df_alpha_ese_Vhat = pd.DataFrame({'ese_alpha_Vhat': ese_alpha_Vhat})
    # df_Vhat = pd.concat([df_ese_Vhat,df_alpha_ese_Vhat],axis =1 )

    df = pd.concat([df_result_W,df_result_Vhat,df_count_W,df_count_Vhat,df_count_iner_iteration,df_ese_W,df_alpha_ese_W,df_ese_Vhat,df_alpha_ese_Vhat],axis =1 )
    # delete unqualified data
    filtered_df = df[~((df['_Result_'] == False) |
                       (df['_Result2_'] == False) |
                    # (abs(df['gamma'] - gamma0) >= 0.5)|
                        (df['iterations_W'] == 101) |
                        (df['iterations_Vhat'] == 101) |
                        (df['iner_iteration'] == 101) |
                       (df['ese_gamma_W'] >= 3)|
                       (df['ese_gamma_Vhat'] >= 3)
                    #    (abs(df['beta1'] + 2) >= 0.5)|
                    #    (abs(df['beta2'] - 2) >= 0.5)
                    ) ]
    filtered_df.reset_index(drop=True, inplace=True)
    print('len(df) After selection：',len(filtered_df),'Proportion of unqualified data',1 - len(filtered_df)/times)
    df_head = filtered_df.head(min([1000,len(filtered_df)]))
    m = df_head.mean()
    summary[f'{n0}_{n1}_{a}_{b1}_{b2}_{p1}'] = m
    summary.to_csv('case control via summary data/sigma_I and sigma w/corollary2.csv')

print(summary)