%% Control of an Under Actuated P-R-R Serial Mechanism to Represent a Backflip
% Unit: m-kg-sec-rad

clear;clc;close all;

tic

n = 3; % # of DoFs
g = 9.81;
% assume robot torso is a rectangle
L = 0.24; % robot torso length
W = 0.24; % robot torso width
H = 0.24; % robot torso height
L_leg = 0.3*ones(1,2); % robot leg link length
R_out = 0.15; % reaction wheel outer radius. Assume reaction wheel is a ring
R_in = 0.12;
reaction_wheel_thickness = 0.03;
m = [0; 7; 3]; % link mass
m_tot = sum(m);
Iyy = [0; m(2)*(L^2 + H^2)/12; 0.5*m(3)*(R_out^2 + R_in^2)]; % link rotary inertia

M = [m_tot,0,0;...
    0,Iyy(2) + Iyy(3),Iyy(3);...
    0,Iyy(3),Iyy(3)]; % inertia matrix
cgterm = [g*m_tot; 0; 0]; % velocity product and gravitational terms

dt = 1e-3; % simulation sampling frequency
T = 1; % total simulation time
N = T/dt + 1; % # of simulation time steps
t = 0:dt:T; % simulation time

q = zeros(n,N);
dq = zeros(n,N);
ddq = zeros(n,N);

u = zeros(n - 1,N); % underactuation
Bu = [1,0; 0,-1,; 0,1];

phase = zeros(1,N); % 0 -> stance phase. 1 -> aerial phase

q_cmd = zeros(n,N); % joint position command
dq_cmd = zeros(n,N);
ddq_cmd = zeros(n,N);
ddq_esm = zeros(n,N); % joint acceleration estimate

q_leg = zeros(2,N); % joint position of planar serial R-R leg
u_leg = zeros(2,N); % joint actuation of planar serial R-R leg
wrench_leg_max = zeros(2,N);
joint_torque_leg_max = 30;

q_2_cmd_end = 0;
q_2_flip = -2*pi;

q(:,1) = [0.1; 0; 0]; % initial joint position
stroke = 0.49;
release_height = q(1,1) + stroke;
spring_stiffness = 700;

flip_time = 0.5;
a_traj = zeros(4,1);

kp = [0,100]; % joint 2 PD control gains
kd = [0,10];
omega_cutoff = 10;

for i_time = 1:N
    % detect phase
    if q(1,i_time) > release_height
        phase(i_time) = 1;
    else
        phase(i_time) = 0;
    end
    
    % plan cubic trajectory after lift-off
    if i_time > 1 && phase(i_time - 1) == 0 && phase(i_time) == 1
        t_traj_start = t(i_time);
        t_traj_end = t_traj_start + flip_time;
        
        t_lift_off = t_traj_start;
        i_time_lift_off = i_time;
        
        T_traj = [1,t_traj_start,t_traj_start^2,t_traj_start^3;...
            1,t_traj_end,t_traj_end^2,t_traj_end^3;...
            0,1,2*t_traj_start,3*t_traj_start^2;...
            0,1,2*t_traj_end,3*t_traj_end^2];
        
        % joint position command at start = 0, at end = 2*pi
        % joint position command at start = 0, at end = 0
        q_2_cmd_end = q(2,i_time) + q_2_flip;
        a_traj = T_traj\[q(2,i_time); q_2_cmd_end; 0; 0]; % cubic trajectory parameters
    end
    
    % detect touch-down
    if i_time > 1 && phase(i_time - 1) == 1 && phase(i_time) == 0
        t_touch_down = t(i_time);
        i_time_touch_down = i_time;
        q_2_flip = -q_2_flip;
    end
    
    % compute joint 2 cubic trajectory
    if phase(i_time) == 1 && t(i_time) < t_traj_end
        q_cmd(2,i_time) = a_traj(1) + a_traj(2)*t(i_time) + a_traj(3)*t(i_time)^2 + a_traj(4)*t(i_time)^3;
        dq_cmd(2,i_time) = a_traj(2) + 2*a_traj(3)*t(i_time) + 3*a_traj(4)*t(i_time)^2;
        ddq_cmd(2,i_time) = 2*a_traj(3) + 6*a_traj(4)*t(i_time);
    else
        q_cmd(2,i_time) = q_2_cmd_end;
    end
    
    % actuate joint 3 using PD
    u(2,i_time) = Iyy(3)*ddq_cmd(2,i_time) + kd(2)*(dq_cmd(2,i_time) - dq(2,i_time)) + kp(2)*(q_cmd(2,i_time) - q(2,i_time));
    u(2,i_time) = -u(2,i_time);
    
    % actuate joint 1 as a spring in stance phase
    if phase(i_time) == 0
        u(1,i_time) = spring_stiffness*(release_height - q(1,i_time));
    else
        u(1,i_time) = 0;
    end
    
    % assume planar serial R-R leg
    % inverse kinematics
    if phase(i_time) == 0
        LHF = q(1,i_time); % distance between hip and foot
        q_leg(2,i_time) = acos( (LHF^2 - L_leg(1)^2 - L_leg(2)^2)/(2*L_leg(1)*L_leg(2)) );
        
        alpha = acos( (L_leg(1)^2 + LHF^2 - L_leg(2)^2)/(2*L_leg(1)*LHF) );
        q_leg(1,i_time) = -atan2(q(1,i_time),0) - alpha;
    else
        q_leg(:,i_time) = q_leg(:,i_time - 1);
    end
    
    % compute geometric Jacobian
    J = [-L_leg(1)*sin(q_leg(1,i_time)) - L_leg(2)*sin(q_leg(1) + q_leg(2)),-L_leg(2)*sin(q_leg(1) + q_leg(2));...
        L_leg(1)*cos(q_leg(1,i_time)) + L_leg(2)*cos(q_leg(1) + q_leg(2)),L_leg(2)*cos(q_leg(1) + q_leg(2))];
    
    % compute actuation torques
    u_leg(:,i_time) = (J')*[0; u(1,i_time)];
    
    % compute max leg wrench
    wrench_leg_max(:,i_time) = inv(J')*joint_torque_leg_max*ones(2,1);
    
    % compute dynamics
    ddq(:,i_time) = M\(Bu*u(:,i_time) - cgterm);
    
    % integrate
    if i_time < N
        dq(:,i_time + 1) = dq(:,i_time) + ddq(:,i_time)*dt;
        q(:,i_time + 1) = q(:,i_time) + dq(:,i_time)*dt;
    end
end

% plotting
label_font_size = 12;

figure
tiledlayout(3,3,'tilespacing','tight','padding','tight')
i_tile_list = [1,2,4,5,7,8];
i_joint_list = [1,1,2,2,3,3];
unit_q = [" [m]"," [rad]"," [rad]"];
unit_dq = [" [m/s]"," [rad/s]"," [rad/s]"];
unit_u = [" [N]", " [Nm]"];
for i_plot = 1:6
    i_tile = i_tile_list(i_plot);
    i_joint = i_joint_list(i_plot);
    nexttile(i_tile)
    
    if i_tile == 1 || i_tile == 4 || i_tile == 7
        plot(t,q(i_joint,:),'k-')
        
        if i_tile == 4
            hold on
            plot_command_q = plot(t,q_cmd(i_joint,:),'r--');
            hold off
        end
        
        ylabel(strcat("Joint Position ",num2str(i_joint),unit_q(i_joint)),'fontsize',label_font_size)
    elseif i_tile == 2 || i_tile == 5 || i_tile == 8
        plot(t,dq(i_joint,:),'k-')
        
        if i_tile == 5
            hold on
            plot_command_dq = plot(t,dq_cmd(i_joint,:),'r--');
            hold off
        end
        
        ylabel(strcat("Joint Velocity ",num2str(i_joint),unit_dq(i_joint)),'fontsize',label_font_size)
    end
    
    grid on
    
    if i_tile >= 7
        xlabel('Time [s]','fontsize',label_font_size)
    end
    
    sgtitle(strcat("Total Mass = ",num2str(m_tot)," kg, ",...
        "Torso Inertia = ",num2str(round(Iyy(2),4)), "kg-m^2, ",...
        "Reaction Wheel Inertia = ",num2str(round(Iyy(3),4)), "kg-m^2"))
    
end

% plot actuation
nexttile(3)
plot(t,u(1,:),'k-')
grid on
ylabel('Joint Actuation 1 [N]','fontsize',label_font_size)

nexttile(6)
plot(t,u(2,:),'k-')
grid on
ylabel('Joint Actuation 2 [Nm]','fontsize',label_font_size)
xlabel('Time [s]','fontsize',label_font_size)

figure
plot(t,u_leg(1,:),'k--',t,u_leg(2,:),'b-')
grid on
ylabel('Serial Leg Joint Actuation [Nm]','fontsize',label_font_size)
xlabel('Time [s]','fontsize',label_font_size)
legend('Hip','Knee')

figure
plot(t,wrench_leg_max(1,:),'k--',t,wrench_leg_max(2,:),'b-')
grid on
ylabel('Serial Leg Max Foot Force [N]','fontsize',label_font_size)
xlabel('Time [s]','fontsize',label_font_size)
legend('Horizontal','Vertical')

% figure
% plot(t,u(2,:),'r--',t,Iyy(3)*(ddq(3,:) + ddq(2,:)),'k-')
% 
% figure
% plot(t,Iyy(2)*dq(2,:) + Iyy(3)*(dq(3,:) + dq(2,:)),'k-')

toc

%% Video Creation
clc;close all;

tic

ground_height = 0;
ground_half_length = 0.75;
ground_half_width = 0.5;
ground_vertices = [[ground_half_length; ground_half_width; ground_height],[ground_half_length; -ground_half_width; ground_height],...
    [-ground_half_length; -ground_half_width; ground_height],[-ground_half_length; ground_half_width; ground_height]];
ceiling_height = 1.8;

plot_line_width = 4;
plot_marker_size = 12;
label_font_size = 16;
tick_label_font_size = 12;
legend_font_size = 16;
title_font_size = 20;

visualization = figure(99);
visualization.WindowState = 'maximized';
video = VideoWriter('Video','MPEG-4');
% simulation frequency = 1000 Hz
% real time frame rate and video recording period combinations:
% 1. 200 fps + 5
% 2. 100 fps + 10
% 3. 50 fps + 20
video.FrameRate = 50;
open(video)
set(visualization,'doublebuffer','on');

tiledlayout(1,2,'tilespacing','tight','padding','tight')

for i_time = 1:20:N
    nexttile(1)
    p_leg_O_knee = [L_leg(1)*cos(q_leg(1,i_time)); L_leg(1)*sin(q_leg(1,i_time))];
    p_leg_O_foot = p_leg_O_knee + [L_leg(2)*cos(q_leg(1,i_time) + q_leg(2,i_time));...
        L_leg(2)*sin(q_leg(1,i_time) + q_leg(2,i_time))];
    
    plot([0,p_leg_O_knee(1),p_leg_O_foot(1)],[0,p_leg_O_knee(2),p_leg_O_foot(2)],'k-')
    axis equal
    
    xlim([-1,1])
    ylim([-1,1])
    
    
    TOC = [[ry(q(2,i_time)),[0; 0; q(1,i_time)]]; [0,0,0,1]]; % CoM configuration in world frame
    TCW = [[ry(q(3,i_time)),[0; (W + reaction_wheel_thickness)/2; 0]]; [0,0,0,1]]; % reaction wheel configuration in CoM frame
    TOW = TOC*TCW;
    
    pOC = getp(TOC);
    pOW = getp(TOW);
    
    % compute robot foot position for video
    if phase(i_time) == 0
        foot_CoM_length = abs(q(1,i_time));
    else
        foot_CoM_length = release_height;
    end
    pOF = TOC*[0; 0; -foot_CoM_length; 1]; % foot position in world frame
    pOH = TOC*[0; 0; 0; 1]; % hip position in world frame
    TOH = [[ry(q(2,i_time)),pOH(1:3)]; [0,0,0,1]]; % CoM configuration in world frame
    
    leg_length = foot_CoM_length;
    pHK_front = [sqrt(L_leg(1)^2 - 0.25*leg_length^2); 0; -0.5*leg_length; 1];
    pHK_back = [-sqrt(L_leg(1)^2 - 0.25*leg_length^2); 0; -0.5*leg_length; 1];
    pOK_front = TOH*pHK_front;
    pOK_back = TOH*pHK_back;
    
    nexttile(2)
    fill3(ground_vertices(1,:),ground_vertices(2,:),ground_vertices(3,:),'g','facealpha',0.4)
    hold on
    DrawCube([L,W,H],zeros(3,1),'k',TOC,0.25); % draw torso
    DrawCube([2*R_out,reaction_wheel_thickness,2*R_out],zeros(3,1),'k',TOW,0.5); % draw reaction wheel
    plot_reaction_wheel_joint = plot3(pOW(1),pOW(2),pOW(3),'bo','markersize',plot_marker_size,'MarkerEdgeColor','b','MarkerFaceColor','b');
    plot3([pOC(1),pOW(1)],[pOC(2),pOW(2)],[pOC(3),pOW(3)],'b--','linewidth',plot_line_width)
    fill_torso_back = DrawCube([0,W,H],[-L/2 - 1e-6; 0; 0],'m',TOC,1);
    fill_reaction_wheel_back = DrawCube([0,reaction_wheel_thickness,2*R_out],[-2*R_out/2 - 1e-6; 0; 0],'m',TOW,1);
    
    plot_CoM = plot3(0,0,q(1,i_time),'ko','markersize',plot_marker_size,'MarkerEdgeColor','k','MarkerFaceColor','k');
    plot_foot = plot3(pOF(1),pOF(2),pOF(3),'ro','markersize',plot_marker_size,'MarkerEdgeColor','r','MarkerFaceColor','r');
    plot_hip = plot3(pOH(1),pOH(2),pOH(3),'bo','markersize',plot_marker_size,'MarkerEdgeColor','b','MarkerFaceColor','b');
    plot_knee_front = plot3(pOK_front(1),pOK_front(2),pOK_front(3),'bo','markersize',plot_marker_size,'MarkerEdgeColor','b','MarkerFaceColor','b');
    plot_knee_back = plot3(pOK_back(1),pOK_back(2),pOK_back(3),'bo','markersize',plot_marker_size,'MarkerEdgeColor','b','MarkerFaceColor','b');
    plot_leg = plot3([pOH(1),pOK_front(1),pOF(1),pOK_back(1),pOH(1)],...
        [pOH(2),pOK_front(2),pOF(2),pOK_back(2),pOH(2)],...
        [pOH(3),pOK_front(3),pOF(3),pOK_back(3),pOH(3)],...
        'k-','linewidth',plot_line_width);
    
    hold off
    axis equal
    grid on
    
    view(45,30)
    xlim(ground_half_length*[-1,1])
    ylim(ground_half_width*[-1,1])
    zlim([0,ceiling_height])
    
    xticks(-ground_half_length:0.25:ground_half_length)
    yticks(-ground_half_width:0.25:ground_half_width)
    zticks(0:0.2:ceiling_height)
    
    ax = gca;
    ax.XAxis.FontSize = tick_label_font_size;
    ax.YAxis.FontSize = tick_label_font_size;
    ax.ZAxis.FontSize = tick_label_font_size;
    
    xlabel('{\itx} [m]','fontsize',label_font_size)
    ylabel('{\ity} [m]','fontsize',label_font_size)
    zlabel('{\itz} [m]','fontsize',label_font_size)
    
    legend([plot_CoM,plot_hip,plot_foot],{'CoM','Joint','Foot'},'fontsize',legend_font_size,'location','north')
    
    title(strcat("Time = ",num2str(1000*round(t(i_time),2))," ms"),'fontsize',title_font_size)
    
    pause(1e-3)
    frame = getframe(visualization);
    drawnow;
    writeVideo(video,frame);
    
end
close(video);


toc

%% Functions
function p = getp(T)
p = T(1:3,4);
end

function R = getr(T)
pR = T(1:3,4);
end

function R = ry(q)
R = [cos(q),0,sin(q);...
    0,1,0;...
    -sin(q),0,cos(q)];
end
