
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

V = 0.00227
Q = 0.001
tau = V / Q

C_A_in = 2000
C_B_in = 0
C_C_in = 0

T = 350

R = 8.314

k0_1 = 1e7
Ea1 = 50000

k0_2 = 5e6
Ea2 = 55000

def k1(T):
    return k0_1 * np.exp(-Ea1 / (R * T))

def k2(T):
    return k0_2 * np.exp(-Ea2 / (R * T))

k1_val = k1(T)
k2_val = k2(T)

Da1 = k1_val * tau
Da2 = k2_val * tau

def cstr_equations(vars, tau_val, T_val):
    C_A, C_B, C_C = vars
    
    k1_local = k0_1 * np.exp(-Ea1 / (R * T_val))
    k2_local = k0_2 * np.exp(-Ea2 / (R * T_val))

    res_A = Q * (C_A_in - C_A) - k1_local * C_A * V
    res_B = -Q * C_B + (k1_local * C_A - k2_local * C_B) * V
    res_C = -Q * C_C + k2_local * C_B * V
    
    return [res_A, res_B, res_C]

def solve_cstr(tau_val, T_val, guess=None):
    if guess is None:
        guess = [C_A_in * 0.5, C_A_in * 0.3, C_A_in * 0.2]
    
    Q_local = V / tau_val
    
    global Q
    Q_original = Q
    Q = Q_local
    
    solution = fsolve(cstr_equations, guess, args=(tau_val, T_val))
    
    Q = Q_original
    
    return solution

C_A, C_B, C_C = solve_cstr(tau, T)

X_A = (C_A_in - C_A) / C_A_in
Y_B = C_B / C_A_in
S_B = C_B / (C_A_in - C_A) if (C_A_in - C_A) > 0 else 0

print(f'Temperature: {T - 273:.1f} degC ({T:.1f} K)')
print(f'Residence time: {tau:.1f} s')
print(f'Da1: {Da1:.3f}, Da2: {Da2:.3f}, Da2/Da1: {Da2/Da1:.3f}')
print(f'C_A: {C_A/1000:.3f} mol/L, C_B: {C_B/1000:.3f} mol/L, C_C: {C_C/1000:.3f} mol/L')
print(f'Conversion: {X_A*100:.1f}%, Yield: {Y_B*100:.1f}%, Selectivity: {S_B*100:.1f}%')

tau_range = np.linspace(1, 100, 200)
C_A_vals = []
C_B_vals = []
C_C_vals = []

for t in tau_range:
    try:
        sol = solve_cstr(t, T)
        C_A_vals.append(sol[0])
        C_B_vals.append(sol[1])
        C_C_vals.append(sol[2])
    except:
        C_A_vals.append(np.nan)
        C_B_vals.append(np.nan)
        C_C_vals.append(np.nan)

B_array = np.array(C_B_vals)
max_idx = np.nanargmax(B_array)
tau_opt = tau_range[max_idx]
B_max = B_array[max_idx] / 1000

plt.figure(figsize=(12, 7))
plt.plot(tau_range, np.array(C_A_vals)/1000, 'b-', linewidth=2, label='Ethanol (A)')
plt.plot(tau_range, np.array(C_B_vals)/1000, 'g-', linewidth=2, label='Ethanal (B)')
plt.plot(tau_range, np.array(C_C_vals)/1000, 'r-', linewidth=2, label='Ethanoic Acid (C)')
plt.axvline(x=tau, color='k', linestyle='--', linewidth=1.5, label=f'tau = {tau:.0f}s')
plt.plot(tau_opt, B_max, 'mo', markersize=10, label=f'tau_opt = {tau_opt:.1f}s, B_max = {B_max:.3f} mol/L')
plt.xlabel('Residence time tau (s)')
plt.ylabel('Concentration (mol/L)')
plt.title('Series Reactions A -> B -> C: Intermediate B peaks at optimal tau')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('ExtensionA_Concentrations.png', dpi=300, bbox_inches='tight')
plt.show()

X_vals = []
Y_vals = []
S_vals = []

for i, t in enumerate(tau_range):
    if not np.isnan(C_A_vals[i]):
        X = (C_A_in - C_A_vals[i]) / C_A_in
        Y = C_B_vals[i] / C_A_in
        S = C_B_vals[i] / (C_A_in - C_A_vals[i]) if (C_A_in - C_A_vals[i]) > 0 else 0
        X_vals.append(X)
        Y_vals.append(Y)
        S_vals.append(S)
    else:
        X_vals.append(np.nan)
        Y_vals.append(np.nan)
        S_vals.append(np.nan)

plt.figure(figsize=(12, 7))
plt.plot(tau_range, X_vals, 'b-', linewidth=2, label='Conversion of A')
plt.plot(tau_range, Y_vals, 'g-', linewidth=2, label='Yield of B')
plt.plot(tau_range, S_vals, 'r--', linewidth=2, label='Selectivity to B')
plt.axvline(x=tau, color='k', linestyle='--', linewidth=1.5, label=f'tau = {tau:.0f}s')
plt.axvline(x=tau_opt, color='m', linestyle=':', linewidth=1.5, label=f'tau_opt = {tau_opt:.1f}s')
plt.xlabel('Residence time tau (s)')
plt.ylabel('Fraction')
plt.title('Conversion, Yield, and Selectivity vs. Residence Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('ExtensionA_Yield_Selectivity.png', dpi=300, bbox_inches='tight')
plt.show()

T_range = np.linspace(320, 400, 50)
Y_B_at_tau = []
S_B_at_tau = []

for T_temp in T_range:
    try:
        sol = solve_cstr(tau, T_temp)
        C_A_temp, C_B_temp, C_C_temp = sol
        Y_temp = C_B_temp / C_A_in
        S_temp = C_B_temp / (C_A_in - C_A_temp) if (C_A_in - C_A_temp) > 0 else 0
        Y_B_at_tau.append(Y_temp)
        S_B_at_tau.append(S_temp)
    except:
        Y_B_at_tau.append(np.nan)
        S_B_at_tau.append(np.nan)

plt.figure(figsize=(12, 7))
plt.plot(T_range - 273, Y_B_at_tau, 'g-', linewidth=2, label='Yield of B')
plt.plot(T_range - 273, S_B_at_tau, 'r--', linewidth=2, label='Selectivity to B')
plt.axvline(x=T - 273, color='k', linestyle='--', linewidth=1.5, label=f'T = {T-273:.1f} degC')
plt.xlabel('Temperature (degC)')
plt.ylabel('Fraction')
plt.title('Temperature Effect on Ethanal Yield and Selectivity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('ExtensionA_Temperature_Effect.png', dpi=300, bbox_inches='tight')
plt.show()

Da1_grid = np.logspace(-1, 2, 50)
Da2_grid = np.logspace(-1, 2, 50)
Da1_mesh, Da2_mesh = np.meshgrid(Da1_grid, Da2_grid)
Y_B_theory = Da1_mesh / ((1 + Da1_mesh) * (1 + Da2_mesh))

plt.figure(figsize=(12, 7))
contour = plt.contourf(Da1_mesh, Da2_mesh, Y_B_theory, levels=20, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('Theoretical yield of B')
plt.plot(Da1, Da2, 'ro', markersize=12, label=f'Da1={Da1:.2f}, Da2={Da2:.2f}')
plt.xlabel('Da1 = k1 * tau')
plt.ylabel('Da2 = k2 * tau')
plt.title('Damköhler Map for Series Reactions A -> B -> C')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ExtensionA_Damkohler_Map.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'Optimal tau: {tau_opt:.1f} s')
print(f'Maximum B concentration: {B_max:.3f} mol/L')
print(f'Maximum B yield: {Y_B_opt*100:.1f}%')