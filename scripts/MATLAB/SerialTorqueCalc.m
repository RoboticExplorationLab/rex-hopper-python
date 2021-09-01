%%
clc;clear;close all;

tic

n = 2;
L = [0.3,0.3];
q_lift_off = deg2rad([-90,0]);

q = sym('q_%d',[n,1],'real');
J(q) = [-L(1)*sin(q(1)) - L(2)*sin(sum(q)),-L(2)*sin(sum(q));...
    L(1)*cos(q(1)) + L(2)*cos(sum(q)),L(2)*cos(sum(q))];

force_lift_off = [0; 240];
torque_lift_off = (J(q_lift_off(1),q_lift_off(2))')*force_lift_off

toc
