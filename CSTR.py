
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint

V = 0.010
D = 0.20
H_jacket = 0.20
A = np.pi * D * H_jacket

Q = 0.001
tau = V / Q
C_A_in = 2000
T_in = 300
T_cool = 300

rho = 1000
Cp = 4180
delta_H = -80000

k0 = 1e8
Ea = 40000
R = 8.314

U = 500
UA = U * A

Q_m3s = Q / 1000
V_m3 = V / 1000

def k(T):
    return k0 * np.exp(-Ea / (R * T))

def C_A_ss(T, tau_val=None):
    if tau_val is None:
        tau_val = tau
    return C_A_in / (1 + k(T) * tau_val)

def Q_gen(T, tau_val=None):
    if tau_val is None:
        tau_val = tau
    return (-delta_H) * k(T) * C_A_ss(T, tau_val) * V_m3

def Q_rem(T, T_cool_val=None, T_in_val=None):
    if T_cool_val is None:
        T_cool_val = T_cool
    if T_in_val is None:
        T_in_val = T_in
    return Q_m3s * rho * Cp * (T - T_in_val) + UA * (T - T_cool_val)

def residual(T, tau_val=None, T_cool_val=None, T_in_val=None):
    if tau_val is None:
        tau_val = tau
    if T_cool_val is None:
        T_cool_val = T_cool
    if T_in_val is None:
        T_in_val = T_in
    return Q_gen(T, tau_val) - Q_rem(T, T_cool_val, T_in_val)

T_ss = fsolve(residual, 320)[0]
k_ss = k(T_ss)
C_A_ss_val = C_A_ss(T_ss)
X_ss = 1 - C_A_ss_val / C_A_in
Da_ss = k_ss * tau

print(f'Reactor volume: {V*1000:.1f} L')
print(f'Residence time: {tau:.1f} s')
print(f'Jacket area: {A:.3f} m^2')
print(f'UA: {UA:.1f} W/K')
print(f'Exit C_A: {C_A_ss_val/1000:.3f} mol/L')
print(f'Exit T: {T_ss-273:.1f} degC ({T_ss:.1f} K)')
print(f'Conversion: {X_ss*100:.1f}%')
print(f'Damköhler Da: {Da_ss:.3f}')

T_range = np.linspace(290, 550, 500)
Q_gen_vals = [Q_gen(T) for T in T_range]
Q_rem_vals = [Q_rem(T) for T in T_range]

plt.figure(figsize=(10, 6))
plt.plot(T_range, Q_gen_vals, 'r-', linewidth=2, label='Q_gen')
plt.plot(T_range, Q_rem_vals, 'b-', linewidth=2, label='Q_rem')
plt.plot(T_ss, Q_gen(T_ss), 'go', markersize=10)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat (W)')
plt.title('Heat Generation vs. Removal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('CSTR_HeatCurves.png', dpi=300, bbox_inches='tight')
plt.show()

tau_range = np.linspace(1, 50, 200)
X_range = []
for t in tau_range:
    T_temp = fsolve(lambda T: residual(T, tau_val=t), 320)[0]
    k_temp = k(T_temp)
    X_temp = (k_temp * t) / (1 + k_temp * t)
    X_range.append(X_temp)

plt.figure(figsize=(10, 6))
plt.plot(tau_range, X_range, 'g-', linewidth=2)
plt.plot(tau, X_ss, 'ro', markersize=8)
plt.xlabel('Residence time tau (s)')
plt.ylabel('Conversion X')
plt.title('Conversion vs. Residence Time')
plt.grid(True, alpha=0.3)
plt.savefig('CSTR_Conversion_vs_Tau.png', dpi=300, bbox_inches='tight')
plt.show()