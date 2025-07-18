import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Physical parameters (all in SI units)
eta = 500e6  # Diffusion coefficient in m²/s (500 km²/s converted to m²/s)
R_sun = 6.96e8  # Solar radius in meters
r = 5 * R_sun  # Radial distance (5 solar radii)
tau = 5.5 * 365.25 * 24 * 3600  # Decay time constant: 5 years in seconds
u0 = 12.5  # Velocity amplitude in m/s
lambda_0 = np.deg2rad(75)  # Latitude threshold in radians
peak_pos_deg = 45
# Numerical parameters - adjusted for stability
lambda_max = 75  # Maximum latitude in degrees
n_points = 100  # Reduced number of grid points for stability
T_years = 11  # Total simulation time in years

# Convert to radians and set up grid
lambda_max_rad = np.deg2rad(lambda_max)
lambda_grid = np.linspace(-lambda_max_rad, lambda_max_rad, n_points)
dlambda = lambda_grid[1] - lambda_grid[0]

print(f"Grid spacing: {np.rad2deg(dlambda):.3f} degrees")
print(f"Diffusion coefficient: {eta/1e6:.0f} × 10⁶ m²/s")
print(f"Decay time: {tau/(365.25*24*3600):.1f} years")
print(f"Velocity amplitude: {u0} m/s")

def velocity_u(lambda_val, u0, lambda_0):
    """
    Meridional velocity u(λ) as given in the equation:
    u(λ) = u0 * sin(πλ/λ0) if |λ| ≤ λ0, otherwise 0
    """
    u = np.zeros_like(lambda_val)
    mask = np.abs(lambda_val) <= lambda_0
    u[mask] = u0 * np.sin(np.pi * lambda_val[mask] / lambda_0)
    return u

def source_term(lambda_val, t):
    """
    Source term S(λ,t) - currently set to zero but structured for future definition.
    This will be treated explicitly since it's a function of λ and t.
    """
    return np.zeros_like(lambda_val)

def create_implicit_matrix(lambda_grid, eta, R_sun, tau, dt):
    """
    Create the matrix for implicit terms with improved stability:
    1. Diffusion term: (η/(R² cos λ)) ∂²(Br cos λ)/∂λ²
    2. Decay term: -Br/τ
    
    Matrix equation: (I - dt*L_impl) * Br^{n+1} = RHS
    """
    n = len(lambda_grid)
    dlambda = lambda_grid[1] - lambda_grid[0]
    
    # Initialize arrays for the tridiagonal matrix
    main_diag = np.zeros(n)
    upper_diag = np.zeros(n-1)
    lower_diag = np.zeros(n-1)
    
    for i in range(1, n-1):  # Interior points
        cos_lambda = np.cos(lambda_grid[i])
        cos_lambda_plus = np.cos(lambda_grid[i+1])
        cos_lambda_minus = np.cos(lambda_grid[i-1])
        
        # Avoid division by zero near poles - use minimum threshold
        cos_lambda = max(abs(cos_lambda), 1e-6) * np.sign(cos_lambda) if cos_lambda != 0 else 1e-6
        
        # Diffusion term: d²(Br cos λ)/dλ² with improved numerical treatment
        # Use a more conservative diffusion coefficient near poles
        pole_factor = min(1.0, abs(cos_lambda) / 0.05)  # Reduce diffusion near poles
        diff_coeff = eta * pole_factor / (R_sun**2 * abs(cos_lambda) * dlambda**2)
        
        # Apply limited diffusion coefficients to prevent instability
        diff_coeff = min(diff_coeff, 1.0 / (tau * 0.1))  # Limit based on decay timescale
        
        main_diag[i] = 1 + dt * (diff_coeff * (cos_lambda_plus + cos_lambda_minus) + 1/tau)
        lower_diag[i-1] = -dt * diff_coeff * cos_lambda_minus
        upper_diag[i] = -dt * diff_coeff * cos_lambda_plus

    
    # Boundary conditions: zero gradient (∂Br/∂λ = 0)
    main_diag[0] = -dt * (-1/tau)  # Only decay at boundary
    main_diag[-1] = -dt * (-1/tau)  # Only decay at boundary
    
    # Create the full implicit operator matrix
    diagonals = [lower_diag, main_diag, upper_diag]
    offsets = [-1, 0, 1]
    L_impl = diags(diagonals, offsets, shape=(n, n), format='csr')
    
    # Return (I - dt*L_impl)
    I = diags([1], [0], shape=(n, n), format='csr')
    return L_impl

def advection_explicit(Br, lambda_grid, u0, lambda_0, R_sun):
    """
    Calculate the advection term explicitly with improved stability:
    (1/(R cos λ)) ∂/∂λ(Br * u(λ) * cos λ)
    """
    n = len(Br)
    dlambda = lambda_grid[1] - lambda_grid[0]
    advection = np.zeros_like(Br)
    
    # Get velocity field
    u = velocity_u(lambda_grid, u0, lambda_0)
    
    # Calculate Br * u(λ) * cos(λ)
    flux = Br * u * np.cos(lambda_grid)
    
    # Calculate ∂/∂λ(Br * u(λ) * cos λ) using central differences
    for i in range(1, n-1):
        cos_lambda = np.cos(lambda_grid[i])
        # Avoid division by zero with minimum threshold
        cos_lambda_safe = max(abs(cos_lambda), 1e-6) * np.sign(cos_lambda) if cos_lambda != 0 else 1e-6
        
        dflux_dlambda = (flux[i+1] - flux[i-1]) / (2 * dlambda)
        advection[i] = -(dflux_dlambda / (R_sun * cos_lambda_safe))
    
    # Boundary conditions for advection (zero gradient)
    advection[0] = 0
    advection[-1] = 0
    
    return advection
    
def normalize_flux(Br, lambda_grid):
    flux = np.trapz(Br * np.cos(lambda_grid), lambda_grid)
    correction = flux / np.trapz(np.cos(lambda_grid), lambda_grid)
    return Br - correction

def rk_imex_step(Br, dt, lambda_grid, eta, R_sun, tau, u0, lambda_0, t):
    """
    Single step of first-order IMEX method with improved numerical stability
    """
    n = len(Br)
    
    # Create implicit matrix (I - dt*L_implicit)
    implicit_matrix = create_implicit_matrix(lambda_grid, eta, R_sun, tau, dt)
    
    # Calculate explicit terms with limiting
    advection = advection_explicit(Br, lambda_grid, u0, lambda_0, R_sun)
    source = source_term(lambda_grid, t)
    
    # Limit the explicit terms to prevent instability
    max_advection = np.max(np.abs(advection))
    if max_advection > 0:
        advection_limit = np.max(np.abs(Br)) / (dt * 10)  # Limit change to 10% per timestep
        if max_advection > advection_limit:
            advection = advection * (advection_limit / max_advection)
    
    explicit_terms = advection + source
    
    # IMEX step: (I - dt*L_impl) * Br^{n+1} = Br^n + dt * E(Br^n)
    rhs = Br + dt * explicit_terms
    
    # Apply boundary conditions to RHS
    if n > 1:
        # Zero gradient boundary conditions
        implicit_matrix[0, :] = 0
        implicit_matrix[0, 0] = 1
        rhs[0] = 0
        
        implicit_matrix[-1, :] = 0
        implicit_matrix[-1, -1] = 1
        rhs[-1] = 0

    
    # Solve the linear system
    try:
        Br_new = spsolve(implicit_matrix, rhs)
    except Exception as e:
        print(f"Linear solve failed: {e}")
        return Br  # Return unchanged if solve fails
    
    # Apply field limiting to prevent runaway values
    max_reasonable_field = 1e-2  # 10 mT maximum
    Br_new = np.clip(Br_new, -max_reasonable_field, max_reasonable_field)
    return Br_new

def create_initial_condition(lambda_grid):
    """
    Create initial condition: two Gaussian peaks of equal magnitude 
    but opposite sign, equidistant from equator on opposite sides.
    """
    # Peak positions (±peak_pos_deg degrees from equator)
    peak_pos_rad = np.deg2rad(peak_pos_deg)
    
    # Gaussian width
    sigma = np.deg2rad(8)  # 8 degrees width
    
    # Amplitude in Tesla (realistic solar magnetic field strength)
    amplitude = 1e-3  # 1 mT
    
    # Positive peak at +30 degrees
    positive_peak = amplitude * np.exp(-(lambda_grid - peak_pos_rad)**2 / (2 * sigma**2))
    
    # Negative peak at -30 degrees  
    negative_peak = -amplitude * np.exp(-(lambda_grid + peak_pos_rad)**2 / (2 * sigma**2))
    
    Br_initial = positive_peak + negative_peak
    
    return Br_initial

# Set up the simulation
print("\nSetting up simulation...")

# Create initial condition
Br_initial = create_initial_condition(lambda_grid)

# Time parameters - adjusted for stability
dt_years = 0.005  # Smaller time step for stability
dt_seconds = dt_years * 365.25 * 24 * 3600
n_steps = int(T_years / dt_years)

print(f"Time step: {dt_years} years ({dt_seconds:.0f} seconds)")
print(f"Number of time steps: {n_steps}")

# Check stability criterion for diffusion
diffusion_stability = dlambda**2 * R_sun**2 / (2 * eta)
print(f"Diffusion stability limit: {diffusion_stability/(365.25*24*3600):.6f} years")
print(f"Using dt: {dt_seconds/(365.25*24*3600):.6f} years")

# Storage for results
save_interval = max(1, n_steps // 100)
saved_indices = list(range(0, n_steps + 1, save_interval))
n_saved = len(saved_indices)

Br_history = np.zeros((n_saved, len(lambda_grid)))
time_history = np.zeros(n_saved)

# Initialize
Br_current = Br_initial.copy()
Br_history[0] = Br_current
time_history[0] = 0

print("\nRunning simulation...")

# Main time loop
save_counter = 1
for step in range(n_steps):
    current_time_seconds = step * dt_seconds
    current_time_years = step * dt_years
    
    # RK-IMEX step
    Br_current = rk_imex_step(Br_current, dt_seconds, lambda_grid, eta, R_sun, tau, u0, lambda_0, current_time_seconds)

    # Normalize total magnetic flux
    #Br_current = normalize_flux(Br_current, lambda_grid)

    # Check for numerical issues
    if np.any(np.isnan(Br_current)) or np.any(np.isinf(Br_current)):
        print(f"Numerical instability detected at step {step}")
        break
    
    # Additional stability check
    if np.max(np.abs(Br_current)) > 1e-1:  # 100 mT threshold
        print(f"Field values too large at step {step}: max = {np.max(np.abs(Br_current))*1000:.1f} mT")
        break
    
    # Save results at specified intervals
    if step + 1 in saved_indices and save_counter < n_saved:
        Br_history[save_counter] = Br_current.copy()
        time_history[save_counter] = (step + 1) * dt_years
        save_counter += 1
    
    # Progress update
    if step % (n_steps // 10) == 0:
        print(f"Progress: {100 * step / n_steps:.1f}% (t = {current_time_years:.2f} years)")

print("Simulation completed!")

# Trim arrays if simulation ended early
if save_counter < n_saved:
    Br_history = Br_history[:save_counter]
    time_history = time_history[:save_counter]

# Create plots with error handling
plt.figure(figsize=(16, 12))

try:
    # Plot 1: Velocity profile
    plt.subplot(3, 3, 1)
    u_profile = velocity_u(lambda_grid, u0, lambda_0)
    plt.plot(np.rad2deg(lambda_grid), u_profile, 'k-', linewidth=2)
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Velocity u(λ) (m/s)')
    plt.title('Meridional Velocity Profile')
    plt.grid(True)

    # Plot 2: Initial and final magnetic field
    plt.subplot(3, 3, 2)
    plt.plot(np.rad2deg(lambda_grid), Br_history[0]*1000, 'b-', linewidth=2, label='Initial')
    if len(Br_history) > 1:
        plt.plot(np.rad2deg(lambda_grid), Br_history[-1]*1000, 'r-', linewidth=2, label=f'Final (t={time_history[-1]:.1f}y)')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Magnetic Field Br (mT)')
    plt.title('Initial vs Final Magnetic Field')
    plt.grid(True)
    plt.legend()

    # Plot 3: Space-time evolution with safety checks
    plt.subplot(3, 3, 3)
    if len(time_history) > 1 and not np.any(np.isnan(Br_history)) and not np.any(np.isinf(Br_history)):
        Lambda, T = np.meshgrid(np.rad2deg(lambda_grid), time_history)
        Br_plot = Br_history * 1000  # Convert to mT
        
        # Check for reasonable values
        if np.max(np.abs(Br_plot)) < 1e6:  # Reasonable threshold
            contour = plt.contourf(T, Lambda, Br_plot, levels=20, cmap='RdBu_r')
            plt.colorbar(contour, label='Magnetic Field Br (mT)')
            plt.xlabel('Time (years)')
            plt.ylabel('Latitude (degrees)')
            plt.title('Magnetic Field Evolution')
        else:
            plt.text(0.5, 0.5, 'Field values too large\nfor plotting', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Magnetic Field Evolution (unstable)')
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor contour plot', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Magnetic Field Evolution')

    # Plot 4: Field evolution at specific latitudes
    plt.subplot(3, 3, 4)
    if len(time_history) > 1:
        latitudes_to_track = [0, 15, 30, 45]  # degrees
        for lat_deg in latitudes_to_track:
            lat_idx = np.argmin(np.abs(np.rad2deg(lambda_grid) - lat_deg))
            field_evolution = Br_history[:len(time_history), lat_idx] * 1000
            plt.plot(time_history, field_evolution, linewidth=2, label=f'λ={lat_deg}°')
    plt.xlabel('Time (years)')
    plt.ylabel('Magnetic Field (mT)')
    plt.title('Field Evolution at Various Latitudes')
    plt.grid(True)
    plt.legend()

    # Plot 5: Total magnetic flux evolution
    plt.subplot(3, 3, 5)
    if len(time_history) > 1:
        total_flux = np.zeros(len(time_history))
        for i in range(len(time_history)):
            integrand = Br_history[i] * np.cos(lambda_grid)
            total_flux[i] = np.trapz(integrand, lambda_grid)
        
        if total_flux[0] != 0:
            plt.plot(time_history, total_flux / total_flux[0], 'purple', linewidth=2)
        else:
            plt.plot(time_history, total_flux, 'purple', linewidth=2)
    plt.xlabel('Time (years)')
    plt.ylabel('Normalized Total Flux')
    plt.title('Total Magnetic Flux Evolution')
    plt.grid(True)

    # Plot 6: Maximum field strength evolution
    plt.subplot(3, 3, 6)
    if len(time_history) > 1:
        max_field = np.max(np.abs(Br_history[:len(time_history)]), axis=1)
        theoretical_decay = np.exp(-time_history * 365.25 * 24 * 3600 / tau)
        plt.semilogy(time_history, max_field / max_field[0], 'brown', linewidth=2, label='Simulation')
        plt.semilogy(time_history, theoretical_decay, 'k--', linewidth=2, label=f'e^(-t/τ), τ={tau/(365.25*24*3600):.0f}y')
    plt.xlabel('Time (years)')
    plt.ylabel('Normalized Max |Br|')
    plt.title('Maximum Field Decay')
    plt.grid(True)
    plt.legend()

    # Plot 7: Multiple time snapshots
    plt.subplot(3, 3, 7)
    if len(time_history) > 4:
        snapshot_indices = [0, len(time_history)//4, len(time_history)//2, 3*len(time_history)//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'black']
        for i, color in zip(snapshot_indices, colors):
            if i < len(time_history):
                plt.plot(np.rad2deg(lambda_grid), Br_history[i]*1000, color=color, 
                        linewidth=2, label=f't={time_history[i]:.1f}y')
    else:
        # Plot all available snapshots
        for i in range(len(time_history)):
            plt.plot(np.rad2deg(lambda_grid), Br_history[i]*1000, 
                    linewidth=2, label=f't={time_history[i]:.1f}y')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Magnetic Field Br (mT)')
    plt.title('Field Snapshots Over Time')
    plt.grid(True)
    plt.legend()

    # Plot 8: Energy evolution
    plt.subplot(3, 3, 8)
    if len(time_history) > 1:
        magnetic_energy = np.zeros(len(time_history))
        mu_0 = 4 * np.pi * 1e-7
        for i in range(len(time_history)):
            energy_density = Br_history[i]**2 / (2 * mu_0)
            integrand = energy_density * np.cos(lambda_grid)
            magnetic_energy[i] = np.trapz(integrand, lambda_grid)
        
        if magnetic_energy[0] > 0:
            plt.semilogy(time_history, magnetic_energy / magnetic_energy[0], 'navy', linewidth=2)
    plt.xlabel('Time (years)')
    plt.ylabel('Normalized Magnetic Energy')
    plt.title('Magnetic Energy Decay')
    plt.grid(True)

    # Plot 9: Advection effect
    plt.subplot(3, 3, 9)
    if len(Br_history) > 0:
        final_advection = advection_explicit(Br_history[-1], lambda_grid, u0, lambda_0, R_sun)
        plt.plot(np.rad2deg(lambda_grid), final_advection*1e6, 'orange', linewidth=2)
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Advection Term (μT/s)')
    plt.title('Advection Effect (Final Time)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/kaggle/working/Numerical_1D_simulation_results_1.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"Plotting error: {e}")
    print("Some plots may not display correctly due to numerical issues")

# Print comprehensive diagnostics
if len(Br_history) > 0:
    print(f"\nDetailed Simulation Results:")
    print(f"Initial max field: {np.max(np.abs(Br_history[0]))*1000:.3f} mT")
    print(f"Final max field: {np.max(np.abs(Br_history[-1]))*1000:.6f} mT")
    if np.max(np.abs(Br_history[0])) > 0:
        print(f"Field decay ratio: {np.max(np.abs(Br_history[-1]))/np.max(np.abs(Br_history[0])):.6f}")
    print(f"Theoretical decay (exp(-t/τ)): {np.exp(-time_history[-1] * 365.25 * 24 * 3600 / tau):.6f}")
    print(f"Diffusion time scale: {R_sun**2/eta/(365.25*24*3600):.2f} years")
    print(f"Decay time scale: {tau/(365.25*24*3600):.1f} years")
    print(f"Max advection velocity: {np.max(np.abs(u_profile)):.1f} m/s")
    print(f"Domain size: ±{lambda_max}° latitude")
    print(f"Grid resolution: {np.rad2deg(dlambda):.3f}°")
    print(f"Simulation completed {len(time_history)} time steps out of {n_steps}")
else:
    print("Simulation failed - no data collected")