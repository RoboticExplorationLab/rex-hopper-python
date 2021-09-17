clc
clear all
Lo = 28; % Free length
y = 18.5;% Deflection
Fs = 1110; % Spring force
K = Fs/y;% K = 1110lb/18.5
Dm = 4.5/1000;
dw = .51/1000 % Wire dia (m)
C = Dm/dw;
G = 82737087516; % Shear Modulus music wire, pa
Na = (dw^4.*G)*((2*C^2)/(1+(2*C^2)))/(8*(Dm)^3*K); % From K=((Dw)^4*G)/(8*(Dm)^3*Na; number of turns 3< Na<15
Ls = Na*dw; % Solid length assumed plane ends
ds = Lo-Ls; % Deformation to solid length
% WF = (4*C+2)/(4*C-3);  Wahl's factor
Tau = (((4*C+2)/(4*C-3))*((8*Fs.*Dm)./((pi).*dw.^3))); % Body shear stress (actual stress)
P = (Lo-2.*Dm)./Na; % Pitch
Cl = (Lo-Ls)/Na; % Coil Clearance
Nt=Na+2; % Total coils
Lambda=atan(P/(pi*Dm))*180/pi; % Has to be less than 12 degrees
CriticalRatio = y/Lo;
Ratio = Lo/Dm;

fprintf(1,'C = %8.5f ',C)
fprintf(1,'\n');
fprintf(1,'K = %8.5f ',K)
fprintf(1,'\n');
fprintf(1,'ds = %8.5f ',ds)
fprintf(1,'\n');
fprintf(1,'Fs = %8.5f ',Fs)
fprintf(1,'\n');
fprintf(1,'Lo = %8.5f ',Lo)
fprintf(1,'\n');
fprintf(1,'dw = %8.5f ',dw)
fprintf(1,'\n');
fprintf(1,'Na = %8.5f ',Na)
fprintf(1,'\n');
fprintf(1,'Ls = %8.5f ',Ls)
fprintf(1,'\n');
fprintf(1,'Tau = %8.5f ',Tau)
fprintf(1,'\n');
fprintf(1,'Clearance > Dw/10 = %8.5f  ',Cl)
fprintf(1,'\n');
fprintf(1,'Total Coils = %8.5f  ',Nt)
fprintf(1,'\n');
fprintf(1,'Pitch = %8.5f  ',P)
fprintf(1,'\n');
fprintf(1,'Pitch Angle < 12 degrees = %8.5f  ',Lambda)
fprintf(1,'\n');
fprintf(1,'Critical Ratio = %8.5f', CriticalRatio)
fprintf(1,'\n');
fprintf(1,'Ratio = %8.5f', Ratio)
fprintf(1,'\n');
