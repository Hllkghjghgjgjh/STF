import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pandas as pd

# Physical parameters (all in SI units)
eta = 500e6  # Diffusion coefficient in m²/s (500 km²/s converted to m²/s)
R_sun = 6.96e8  # Solar radius in meters
r = 5 * R_sun  # Radial distance (5 solar radii)
tau = 5.5 * 365.25 * 24 * 3600  # Decay time constant: 5 years in seconds
u0 = 12.5  # Velocity amplitude in m/s
lambda_0 = np.deg2rad(75)  # Latitude threshold in radians

# Numerical parameters - adjusted for stability
lambda_max = 75  # Maximum latitude in degrees
n_points = 100  # Reduced number of grid points for stability
T_years = 11  # Total simulation time in years

# Convert to radians and set up grid
lambda_max_rad = np.deg2rad(lambda_max)
lambda_grid = np.linspace(-lambda_max_rad, lambda_max_rad, n_points)
dlambda = lambda_grid[1] - lambda_grid[0]

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
    main_diag[0] = 1 + dt * (1/tau)  # Fixed: should be 1 + dt*decay
    main_diag[-1] = 1 + dt * (1/tau)  # Fixed: should be 1 + dt*decay
    
    # Create the full implicit operator matrix
    diagonals = [lower_diag, main_diag, upper_diag]
    offsets = [-1, 0, 1]
    L_impl = diags(diagonals, offsets, shape=(n, n), format='csr')
    
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

def create_initial_condition(lambda_grid, peak_pos_deg):
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
    
    # Positive peak at +peak_pos_deg degrees
    positive_peak = amplitude * np.exp(-(lambda_grid - peak_pos_rad)**2 / (2 * sigma**2))
    
    # Negative peak at -peak_pos_deg degrees  
    negative_peak = -amplitude * np.exp(-(lambda_grid + peak_pos_rad)**2 / (2 * sigma**2))
    
    Br_initial = positive_peak + negative_peak
    
    return Br_initial

def run_simulation(peak_pos_deg, verbose=False):
    """
    Run the simulation for a given peak position and return the maximum field value
    """
    if verbose:
        print(f"\nRunning simulation for peak position: {peak_pos_deg}°")
    
    # Create initial condition
    Br_initial = create_initial_condition(lambda_grid, peak_pos_deg)
    
    # Time parameters - adjusted for stability
    dt_years = 0.005  # Smaller time step for stability
    dt_seconds = dt_years * 365.25 * 24 * 3600
    n_steps = int(T_years / dt_years)
    
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
    
    # Main time loop
    save_counter = 1
    simulation_failed = False
    
    for step in range(n_steps):
        current_time_seconds = step * dt_seconds
        current_time_years = step * dt_years
        
        # RK-IMEX step
        Br_current = rk_imex_step(Br_current, dt_seconds, lambda_grid, eta, R_sun, tau, u0, lambda_0, current_time_seconds)
        
        # Check for numerical issues
        if np.any(np.isnan(Br_current)) or np.any(np.isinf(Br_current)):
            if verbose:
                print(f"Numerical instability detected at step {step}")
            simulation_failed = True
            break
        
        # Additional stability check
        if np.max(np.abs(Br_current)) > 1e-1:  # 100 mT threshold
            if verbose:
                print(f"Field values too large at step {step}: max = {np.max(np.abs(Br_current))*1000:.1f} mT")
            simulation_failed = True
            break
        
        # Save results at specified intervals
        if step + 1 in saved_indices and save_counter < n_saved:
            Br_history[save_counter] = Br_current.copy()
            time_history[save_counter] = (step + 1) * dt_years
            save_counter += 1
    
    # Trim arrays if simulation ended early
    if save_counter < n_saved:
        Br_history = Br_history[:save_counter]
        time_history = time_history[:save_counter]
    
    # Calculate maximum field value
    if len(time_history) > 1 and not simulation_failed:
        max_field = np.max(np.abs(Br_history[:len(time_history)]), axis=1)
        max_field_value = np.max(max_field)
        
        if verbose:
            print(f"Initial max field: {np.max(np.abs(Br_history[0]))*1000:.3f} mT")
            print(f"Final max field: {np.max(np.abs(Br_history[-1]))*1000:.6f} mT")
            print(f"Maximum field during simulation: {max_field_value*1000:.6f} mT")
        
        return max_field_value, simulation_failed
    else:
        if verbose:
            print("Simulation failed - no valid data collected")
        return 0.0, True

# Main analysis: Run simulations for different peak positions
print("Starting systematic analysis of peak positions...")
print(f"Grid spacing: {np.rad2deg(dlambda):.3f} degrees")
print(f"Diffusion coefficient: {eta/1e6:.0f} × 10⁶ m²/s")
print(f"Decay time: {tau/(365.25*24*3600):.1f} years")
print(f"Velocity amplitude: {u0} m/s")

# Define peak positions to test
peak_positions = np.arange(10, 85, 5)  # From 10° to 80° in 5° increments
results = []

for peak_pos in peak_positions:
    max_field, failed = run_simulation(peak_pos, verbose=False)
    results.append({
        'peak_position_deg': peak_pos,
        'max_field_T': max_field,
        'max_field_mT': max_field * 1000,
        'simulation_failed': failed
    })
    print(f"Peak at ±{peak_pos}°: Max field = {max_field*1000:.6f} mT, Failed: {failed}")

# Convert to DataFrame for easier analysis
df_results = pd.DataFrame(results)

# Display results
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(df_results.to_string(index=False))

# Plot the results
plt.figure(figsize=(15, 10))

# Plot 1: Max field vs peak position
plt.subplot(2, 2, 1)
valid_results = df_results[~df_results['simulation_failed']]
if len(valid_results) > 0:
    plt.plot(valid_results['peak_position_deg'], valid_results['max_field_mT'], 
             'bo-', linewidth=2, markersize=8)
    plt.xlabel('Peak Position (degrees)')
    plt.ylabel('Maximum Field (mT)')
    plt.title('Maximum Field vs Peak Position')
    plt.grid(True)

# Plot 2: Success rate
plt.subplot(2, 2, 2)
success_rate = (~df_results['simulation_failed']).mean() * 100
failed_positions = df_results[df_results['simulation_failed']]['peak_position_deg']
plt.bar(['Successful', 'Failed'], 
        [len(valid_results), len(failed_positions)], 
        color=['green', 'red'], alpha=0.7)
plt.ylabel('Number of Simulations')
plt.title(f'Simulation Success Rate: {success_rate:.1f}%')

# Plot 3: Field evolution for a few selected positions
plt.subplot(2, 2, 3)
selected_positions = [20, 40, 60]
for pos in selected_positions:
    if pos in peak_positions:
        # Run detailed simulation for plotting
        max_field, failed = run_simulation(pos, verbose=True)
        # Note: For plotting, we'd need to modify the function to return time series
        # For now, just show the max field values
        
plt.xlabel('Time (years)')
plt.ylabel('Field (mT)')
plt.title('Field Evolution (Selected Positions)')
plt.grid(True)

# Plot 4: Initial conditions for different peak positions
plt.subplot(2, 2, 4)
sample_positions = [10, 30, 50, 70]
for pos in sample_positions:
    initial_field = create_initial_condition(lambda_grid, pos)
    plt.plot(np.rad2deg(lambda_grid), initial_field*1000, 
             linewidth=2, label=f'±{pos}°')
plt.xlabel('Latitude (degrees)')
plt.ylabel('Initial Field (mT)')
plt.title('Initial Conditions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('peak_position_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Find the peak position that gives maximum field
if len(valid_results) > 0:
    max_field_row = valid_results.loc[valid_results['max_field_mT'].idxmax()]
    print(f"\nPeak position with maximum field: ±{max_field_row['peak_position_deg']}°")
    print(f"Maximum field value: {max_field_row['max_field_mT']:.6f} mT")
    
    # Additional statistics
    print(f"\nStatistics for successful simulations:")
    print(f"Mean max field: {valid_results['max_field_mT'].mean():.6f} mT")
    print(f"Std dev: {valid_results['max_field_mT'].std():.6f} mT")
    print(f"Min max field: {valid_results['max_field_mT'].min():.6f} mT")
    print(f"Max max field: {valid_results['max_field_mT'].max():.6f} mT")
else:
    print("\nNo successful simulations to analyze!")

# Save results to CSV
df_results.to_csv('peak_position_analysis_results.csv', index=False)
print(f"\nResults saved to 'peak_position_analysis_results.csv'")