clc
clear all

% From Shigley's Mechanical Design

%--------------------decision variables-----------------------------------%
E = 207*10^9; % Tensile Modulus, pa, music steel
Sigma_max = 1585794800; % max stress, pa

D_pin = 14/1000; % Pin diameter (m)
L1 = 0.05; % leg one radial length (zero if they're bent axially)
L2 = 0.05; % leg two radial length (zero if they're bent axially)
theta_t_d = 60; % required angular deflection, degrees
D_inner = 15/1000; % Spring inner dia
% dw = .51/1000; % wire dia
dw_guess = 5/1000;
T = 2; % Spring torque required, Nm
beta = 180; % partial turn angle of end, in degrees
Nb_integer = 4; % body spring full turns
%-------------------------------------------------------------------------%
theta_t = theta_t_d*pi/180; % converting to radians
Np = beta/360;  % partial turn present in coil body
Nb = Nb_integer + Np;  % number of body turns
Na_guess = Nb+(L1+L2)/(3*pi*(D_inner+dw_guess));
dw = (64*T*(D_inner+dw_guess)*Na_guess/(E*theta_t))^(1/4); % estimate of 
% good wire diameter,starting with baseline.
% This should be adjusted to whatever is close
% dw = 0.51/1000; 
% dw = dw*1.5; % adjust for safety factor
D = D_inner+dw; % Spring mean dia
C = D/dw; % Spring Index
Na = Nb + (L1+L2)/(3*pi*D); % number of active turns
% theta_t = 64*T*D*Na/(E*dw^4) % total angular deflection
k = (dw^4)*E/(64*D*Na); % spring rate
theta_c = 10.8*T*D*Nb/(E*dw^4);  % angular deflection of the spring, turns
D_prime = Nb*D/(Nb + theta_c);  % new diameter after windup
Di_prime = D_prime-dw;  % windup inner diameter
delta = Di_prime-D_pin;  % diametral clearance with pin
% T_max = k*theta_t;  % max torque output
Ki = (4*C^2-C-1)/(4*C*(C-1)); % Stress Concentration Factor
Sigma = Ki*32*T/(pi*dw^3); % Bending Stress

fprintf(1,'Wire Diameter = %8.5f mm', dw*1000)
fprintf(1,'\n');
fprintf(1,'Mean Diameter = %8.5f mm', D*1000)
fprintf(1,'\n');
fprintf(1,'Outer Diameter = %8.5f mm', (D+dw)*1000)
fprintf(1,'\n');
fprintf(1,'Number of active turns = %8.5f', Na)
fprintf(1,'\n');
fprintf(1,'Spring Rate = %8.5f  Nm/radian', k)
fprintf(1,'\n');
fprintf(1,'Spring Rate = %8.5f  Nmm/deg', k*1000*(180/pi))
fprintf(1,'\n');
fprintf(1,'Windup Inner Diameter = %8.5f mm', Di_prime*1000)
fprintf(1,'\n');
fprintf(1,'Diametral Clearance = %8.5f mm', delta*1000)
fprintf(1,'\n');
fprintf(1,'Normal Stress Sigma = %8.5f  Pa', Sigma)
fprintf(1,'\n');
if Sigma > Sigma_max
  fprintf(1, 'Caution: max stress exceeded by %8.5f times!', Sigma/Sigma_max)
  fprintf(1,'\n');
elseif Sigma > Sigma_max/2
  fprintf(1, 'Caution: stress exceeds safety factor of 2!', Sigma/Sigma_max)
  fprintf(1,'\n');
end