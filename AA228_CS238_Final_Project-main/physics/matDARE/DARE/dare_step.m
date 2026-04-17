function [pressure, velocity, temperature, mach_mixture, x_norm, success] = dare_step(equivalence_ratio, mach_flight, altitude)
% DARE_STEP  Run one full DARE scramjet simulation and return key state quantities.
%
%   Inputs:
%     equivalence_ratio  - combustor equivalence ratio (fuel control input)
%     mach_flight        - freestream Mach number (nominally ~6 for scramjet)
%     altitude           - flight altitude [km]
%
%   Outputs:
%     pressure           - nozzle exit static pressure [Pa]
%     velocity           - nozzle exit velocity [m/s]
%     temperature        - nozzle exit temperature [K]
%     mach_mixture       - mixture Mach number at solver exit
%     x_norm             - normalised solver progress: x_exit / (comb_len + nozzle_len)
%                          1.0 = full run completed, <1 = early termination
%     success            - 1 if run reached full nozzle length, 0 otherwise

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, 'StandardAtm'));
addpath(fullfile(base, 'sundialsTB'));
addpath(genpath(fullfile(base, 'sundialsTB')));
addpath(fullfile(base, 'Functions'));
addpath(fullfile(base, 'Examples', 'Scramjet', '1'));  % non-interactive nozzleModule

global Ru gamma phi_stoch engineMode Mach_Mixture

Ru         = 8.314;
gamma      = 1.4;
phi_stoch  = 0.0283;
engineMode = 'SCRAM';

% Flight and atmospheric conditions
[T_Inf, P_Inf, rho_Inf] = atmos(altitude);
atmCond{1} = [T_Inf, P_Inf, rho_Inf];
atmCond{2} = [0.21, 0.79];  % O2, N2

flightComb = [mach_flight, altitude];

% Engine geometry matching Scramjet Example 1
inCond{1} = 2;    % inlet exit Mach
inCond{2} = 6;    % truncation angle
inCond{3} = 4.86; % inlet exit area [m^2]

comb_len   = 8;
nozzle_len = 20;
total_len  = comb_len + nozzle_len;

combCond{1} = comb_len;
combCond{2} = 4;
combCond{3} = 'circular';
combCond{4} = fullfile(base, 'Dependencies', 'area_duct_profile.mat');
combCond{5} = equivalence_ratio;
combCond{6} = fullfile(base, 'Dependencies', 'MW_H2_Air.mat');

nozzCond{1} = nozzle_len;
nozzCond{2} = 0.5;
nozzCond{3} = fullfile(base, 'Dependencies', 'area_duct_profile.mat');
nozzCond{4} = 15;

% Default outputs for failure cases
pressure     = P_Inf;
velocity     = 0;
temperature  = T_Inf;
mach_mixture = 0;
x_norm       = 0;
success      = 0;

try
    [inOut]   = intakeModule(inCond, flightComb, atmCond);
    [isoOut]  = isolatorModule(inOut);
    [combOut] = combustorModule(isoOut, combCond, atmCond);
    [nozzOut] = nozzleModule(combOut, nozzCond, combCond, atmCond);

    x_exit       = nozzOut{2};
    y_exit       = nozzOut{3};
    pressure     = y_exit(3);
    velocity     = y_exit(2);
    temperature  = y_exit(4);
    mach_mixture = Mach_Mixture;
    x_norm       = x_exit / total_len;
    success      = double(x_exit >= total_len);
catch
    % Return defaults on solver failure
end
end
