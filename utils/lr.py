'''
Author: sy
Date: 2021-03-31 15:47:30
LastEditors: pida0
LastEditTime: 2022-07-12 16:46:23
Description: file content
'''
import math

def lambda_warm_up(epoch):
    if epoch <= 6:
        lr_l = 1.0 * (epoch + 1) / 7
        # lr_l=-0.2*epoch+3.15
    elif 6 < epoch <= 10:
        lr_l = 0.1  # 0.15
    elif 10 < epoch <= 37:
        lr_l = 0.01  # 0.15
#
    elif 37 < epoch <= 60:
        lr_l = 0.01 #0.1
    else:
        lr_l = 0.01

    return lr_l

def lambda_warm_up_2(epoch):
    if epoch <= 6:
        lr_l = 1.5 * (epoch + 1) /7
        # lr_l=-0.2*epoch+3.15
    elif 6< epoch <= 15:
        lr_l = 1.5 #3 # 0.15
    elif 15< epoch <= 35:
        lr_l = 0.5#1.5 # 1.5
    elif 35< epoch <= 50:
        lr_l = 0.15#0.75 # 0.15
#
    elif 50 < epoch <= 60:
        lr_l = 0.01#0.5 #0.5
    else:
        lr_l = 0.01
    return lr_l


if __name__ == '__main__':
    base = 0.0001
    lr = []
    ep = []
    for e in range(100):
        lr.append(lambda_warm_up(e)*base)
        ep.append(e)

    import matplotlib.pyplot as plt
    # print(lr[60])
    plt.plot(ep, lr)
    plt.show()