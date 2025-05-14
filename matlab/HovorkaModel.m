function xdot = HovorkaModel(t,x,U,D,par) 
% Chemical reactions in Robertson's chemical reaction problem
%
% 
% Input parameters:
BW=par(1);              % body weight
M_WG=par(2);            % glucose conversion to mmol

%constants
k_12=0.066;
k_a1=0.006;
k_a2=0.06;
k_a3=0.03;
k_e=0.138;
tauD=40;
tauS=55;
A_G=0.8;

k_b1=k_a1*51.2e-4;
k_b2=k_a2*8.2e-4;
k_b3=k_a3*520e-4;

V_I=0.12*BW;
V_G=0.16*BW;
EGP_0=0.0161*BW;
F_01=0.0097*BW;

U_G=x(2)/tauD;
U_I=x(4)/tauS;


% Glucose consumption
G=x(5)/V_G;
if G>=4.5
    F_01c=F_01;
else 
    F_01c=F_01*(G/4.5);
end 

if G>=9
    F_R=0.003*(G-9)*V_G;
else 
    F_R=0;
end 

% Runtime pre calculations
tempD=x(1)/tauD;
tempS=x(3)/tauS;

% Allocating space for x' solution
xdot = zeros(10,1);

% Ddot equations
% D(1)=x(1)
% D(2)=x(2)
xdot(1) = A_G*1000/M_WG*D-tempD; 
xdot(2) = tempD-U_G;

% Sdot equations
% S(1)=x(3)
% S(2)=x(4)
xdot(3) = U-tempS;
xdot(4) = tempS-U_I;

% Qdot equations
% Q(1)=x(5)
% Q(2)=x(6)
xdot(5) = U_G-F_01c-F_R-x(1)*x(5)+k_12*x(6)+EGP_0*(1-x(10));
xdot(6) = x(1)*x(5)-k_12*x(6)-x(2)*x(6);

% Idot equation
% I(1)=x(7)
xdot(7)=U_I/V_I-k_e*x(7);

% Xdot equations
xdot(8) = -k_a1*x(8)+k_b1*x(7);
xdot(9) = -k_a2*x(9)+k_b2*x(7);
xdot(10) = -k_a3*x(10)+k_b3*x(7);



end

