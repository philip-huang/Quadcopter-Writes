#!/usr/bin/env python
from __future__ import division

import numpy as np 
import scipy
import json
import skimage.draw, skimage.io
import matplotlib.pyplot as plt
import scipy.interpolate
import csv
import os
 

def load(filename):
    with open(filename, 'r') as f:
        annot = json.load(f)
        annot = list(annot.values())
        annot = annot[0]
        im_name = annot['filename']
        im = skimage.io.imread(im_name) 
        shapes = [r['shape_attributes'] for r in annot['regions']]

        return im, shapes

def dist(x, y):
    return np.sqrt(np.square(x[0] - y[0]) + np.square(x[1]-y[1]))

def speed_lookup(theta):
    i = 0
    speed_scale = [(170, 1), (160, 0.8), (150, 0.65), (140, 0.5), (130, 0.35), (120, 0.2)]
    while (theta < speed_scale[i][0]):
        i += 1
        if (i == len(speed_scale)):
            return 0.0
    return speed_scale[i][1]    
    

def get_curvature(a, b, c):
    l1 = dist(a, b)
    l2 = dist(b, c)
    l3 = dist(c, a)
    if (l1 == 0 or l2 == 0):
        return 180
    costheta = (-l3 * l3 + l2 * l2 + l1 * l1) / (2 * l1 * l2)
    if (costheta < -1):
        costheta = -1
    if (costheta > 1):
        costheta = 1
    theta = np.arccos(costheta) 

    return np.rad2deg(theta)

def transform(x, y,im_height = 340, im_width = 340,  actual_width = 2.0, actual_height = 2.0): 
    # Center data
    x -= (im_width / 2)
    y -= (im_height / 2)
    # Flip y
    y = -y
    # Scale
    x = x * actual_width / im_width 
    y = y * actual_height / im_height

    return x, y

def draw(im, shapes):
    width, height, _ = im.shape
    actual_width = 2
    actual_height = 2
    trajs = []


    for i, p in enumerate(shapes):
        t_s = []
        x_s = []
        y_s = []

        t_now = 0
        prev_speed = 0
        prev_x = 0
        prev_y = 0
        if p['name'] == 'polyline':
            for j in range(len(p['all_points_x'])):
                x, y = transform(p['all_points_x'][j], p['all_points_y'][j]) 
                if (j == 0): 
                    print(i, x, y)
                    t_s.append(0)  
                    x_s.append(x)
                    y_s.append(y)
                    t_now = 0                
                    prev_x = x
                    prev_y = y
                    continue

                if (j == len(p['all_points_x']) - 1):         
                    speed = 0  
                
                else:
                    x2, y2 = transform(p['all_points_x'][j + 1], p['all_points_y'][j + 1]) 
                    theta = get_curvature((prev_x, prev_y), (x, y), (x2, y2))
                    speed = speed_lookup(theta)
                
                d = dist((prev_x, prev_y), (x, y))
                if d == 0:
                    continue
                x_s.append(x)
                y_s.append(y)
                duration = d / (prev_speed + speed) * 2  
                t_now += duration
                t_s.append(t_now)
                

                prev_x = x
                prev_y = y
                prev_speed = speed 
            trajs.append([t_s, x_s, y_s])
    return trajs

def tocsv(name, t_s, ppx, ppy, plane='xy'):
    length = ppx.c.shape[1]
    a = np.zeros((length, 8 * 4 + 1))
    a[:, 0] = np.diff(t_s)
    if plane=='xy':
        a[:, 1:5] = ppx.c[::-1].T
        a[:, 9:13] = ppy.c[::-1].T
    elif plane=="xz":
        a[:, 1:5] = ppx.c[::-1].T
        a[:, 17:21] = ppy.c[::-1].T
    elif plane=="yz":
        a[:, 9:13] = ppx.c[::-1].T
        a[:, 17:21] = ppy.c[::-1].T
    else:
        print("invalid plane type")
        return 
    
    np.savetxt(name, a, delimiter=",", header='Duration, \
    x^0, x^1, x^2, x^3, x^4, x^5, x^6, x^7, \
    y^0, y^1, y^2, y^3, y^4, y^5, y^6, y^7, \
    z^0, z^1, z^2, z^3, z^4, z^5, z^6, z^7, \
    yaw^0, yaw^1, yaw^2, yaw^3, yaw^4, yaw^5, yaw^6, yaw^7', comments='')
    

def fit(trajs):
    for traj_id, traj in enumerate(trajs):
        t_s, x_s, y_s = traj
        duration = t_s[-1]
        num = len(t_s)
        xdot_s = []
        ydot_s = []
        for i in range(num):
            if (i == 0 or i == num - 1):
                xdot_s.append(0)
                ydot_s.append(0)
                continue 
            xdot1 = (x_s[i] - x_s[i-1])/(t_s[i] - t_s[i-1])
            ydot1 = (y_s[i] - y_s[i-1])/(t_s[i] - t_s[i-1])
            xdot2 = (x_s[i+1] - x_s[i])/(t_s[i+1] - t_s[i])
            ydot2 = (y_s[i+1] - y_s[i])/(t_s[i+1] - t_s[i]) 
            xdot_s.append((xdot1 + xdot2) / 2)
            ydot_s.append((ydot1 + ydot2) / 2)
         
        ppx = scipy.interpolate.CubicSpline(t_s, x_s, bc_type='clamped')
        ppy = scipy.interpolate.CubicSpline(t_s, y_s, bc_type='clamped')
        #tocsv('traj{}.csv'.format(traj_id + 1), t_s, ppx, ppy, plane='xy') 
        tt = np.arange(0, duration + 0.01, 0.01) 
        xt = ppx(tt, 0)
        xdott = ppx(tt, 1)
        yt = ppy(tt, 0)
        ydott = ppy(tt, 1)
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(t_s, x_s, 'o', tt, xt)
        ax.set_title('x') 
        plt.grid()
        ax = fig.add_subplot(223)
        ax.plot(t_s, xdot_s, 'o', tt, xdott)
        ax.set_title('xdot')
        plt.grid()
        ax = fig.add_subplot(222)
        ax.plot(t_s, y_s, 'o', tt, yt)
        ax.set_title('y')  
        plt.grid()
        ax = fig.add_subplot(224)
        ax.plot(t_s, ydot_s, 'o', tt, ydott)
        ax.set_title('ydot')
        plt.grid()
        plt.show()
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        plt.plot(x_s, y_s, 'o', xt, yt)
        ax.set_title("xy")
        plt.grid()
        plt.show()

def plot(trajs):
    for traj in trajs:
        t_s, x_s, y_s = traj
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.plot(t_s, x_s)
        ax.plot(t_s, y_s)
        plt.show()  

def wp2csv(trajs, plane='xy', z_base_height = 0.5): 
    for i, traj in enumerate(trajs):
        _, x_t, y_t = traj
        a = np.zeros((len(x_t), 3), dtype=float)
        if plane == 'xy':
            a[:, 0] = x_t
            a[:, 1] = y_t
        elif plane == 'xz':
            a[:, 0] = x_t
            a[:, 2] = np.array(y_t) + z_base_height
        elif plane == 'yz':
            a[:, 1] = x_t
            a[:, 2] = np.array(y_t) + z_base_height
        np.savetxt(os.path.join('image_data', 'traj{}.csv'.format(i + 1)), a, delimiter=',')
        

im, shapes = load(os.path.join('image_data', 'via_region_data (1).json') 
trajs = draw(im, shapes)   
wp2csv(trajs, plane='xz')
