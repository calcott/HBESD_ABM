# =====================================================
# HYBRID BAYESIAN-EVOLUTIONARY STRUCTURAL DYNAMICS (HBESD)
# Agent-Based Model - Core Simulation Engine
# =====================================================

# =====================================================
# SECTION 1: CONTROL PANEL
# =====================================================
# All global constants, thresholds, and initialization ranges live here.
# DO NOT hardcode these values anywhere else in the simulation.

### COMPUTATIONAL PARAMETERS
MAX_WORKERS       = None            # Set to an int (e.g., 6) or None to auto-detect and use all available CPU cores

### NETWORK PARAMETERS
N_AGENTS          = 50000           # Total number of nodes in the simulation
TOPOLOGY          = "small_world"   # Structural layout
CONNECTION_DENSITY = 0.05           # Watts-Strogatz rewiring probability (p)
AVG_CONNECTIONS   = 6               # Watts-Strogatz k (average neighbors)
RANDOM_SEED       = 42              # Seed for strict reproducibility

### SIMULATION EXECUTION
TOTAL_TICKS       = 100             # Number of time steps to run
BURN_IN_TICKS     = 10              # Ticks before any shocks are applied to let the network stabilize

### SHOCK PARAMETERS (E_k)
SHOCK_TICK        = 10              # Time step at which the macro shock hits
SHOCK_EPICENTER_PCT = 0.05          # Fraction of nodes directly hit by shock (5%)
SHOCK_MAGNITUDE   = 0.999           # E_k value injected at shock tick
SHOCK_DURATION    = 1               # How many ticks the direct shock lasts

### AGENT INITIALIZATION RANGES (Uniform Distributions)
YIELD_THRESHOLD_RANGE     = (1.5, 2.5)    # phi*_i: Absolute breaking point
INITIAL_CAPACITY_RANGE    = (0.8, 1.2)    # C_i(0): Starting elastic capacity
AGENCY_RANGE              = (0.3, 0.9)    # A_i(0): Starting agency scalar
PARASITIC_LOAD_RANGE      = (0.02, 0.08)  # gamma_i: Passive drain rate
INTEGRATION_EFFICIENCY_RANGE = (0.1, 0.3) # kappa_T: Transformation efficiency
STRUCTURAL_ENTROPY_RANGE  = (0.01, 0.05)  # lambda_i: Prior decay rate
COG_SATURATION_RANGE      = (0.5, 1.5)    # omega_i: Yerkes-Dodson saturation limit
INITIAL_PRIOR_RANGE       = (0.40, 0.60)  # P_i(0): Starting prior expectation
THETA_MAX                 = 0.9           # Maximum metabolic conversion efficiency

### HYPOTHESIS SPACE RANGES & SCALARS
H_MAX_RANGE               = (10.0, 20.0)  # H_max: Maximum heuristic slots
H_USED_INIT_RANGE         = (5.0, 10.0)   # H_used(0): Initial occupied slots
LAMBDA_A_MAX_RANGE        = (0.05, 0.15)  # lambda_A,max: Max active update rate
# Option 2 Scaling Factors
H_COMPRESSION_SCALE       = 2.0           # Max slots freed per Transformation
H_EXPANSION_SCALE         = 1.5           # Slots consumed per unit of Strain during Internal Removal

### STRATEGY THRESHOLDS (Coefficients of Yield Threshold phi*)
C_TOL   = 0.15   # Homeostatic tolerance: phi < C_TOL * phi* -> No action
C_ACC   = 0.45   # Acceptance ceiling: C_TOL * phi* <= phi < C_ACC * phi*

### TRANSFORMATION GATE PARAMETERS
AGENCY_ACTIVATION_THRESHOLD = 0.7  # A*: Minimum agency to attempt Transformation
CAPACITY_ACTIVATION_FLOOR   = 0.2  # C*: Minimum absolute capacity to fund Transformation

### TEMPORAL DYNAMICS
ALPHA             = 0.2   # Anxiety coefficient: weight of anticipated future strain
RHO               = 0.3   # Hope coefficient: weight of anticipated future capacity
BETA              = 0.5   # Prior sensitivity: responsiveness to new information
P_BASELINE        = 0.5   # Long-run prior anchor (neutral expectation)

### EMERGENT DYNAMICS (EXTENSION TOGGLE)
ENABLE_EMERGENT_DYNAMICS   = False       # Set False to preserve baseline preprint physics
RHO_RANGE                  = (0.1, 0.4)  # rho_i(0): Starting Hope coefficient
RHO_OBSERVATION_BOOST      = 0.05        # rho increase when observing Transformation
CAPACITY_SURPLUS_THRESHOLD = 1.5         # C_i above this triggers altruistic redistribution
ALTRUISM_TRANSFER_RATE     = 0.1         # Fraction of surplus shared across connections

### NETWORK DYNAMICS
DELTA_TAU         = 0.05  # Amount tau_ij is reduced per External Removal event
DELTA_W           = 0.1   # Amount channel weight is reduced per Internal Removal event
ETA               = 0.1   # Strain attenuation factor in propagation equation

### MULTIPLEX CHANNEL WEIGHTS (Must sum to 1.0)
W_STRUCTURAL_INIT = 0.33  # Initial weight: Structural Channel (macro)
W_IDENTITY_INIT   = 0.34  # Initial weight: Identity/Bounding Channel
W_LOCAL_INIT      = 0.33  # Initial weight: Local Systemic Channel (micro)

### OUTPUT AND LOGGING
OUTPUT_DIR        = "./hbesd_output"
SAVE_FIGURES      = True
SAVE_DATA_CSV     = True
LOG_INTERVAL      = 10    # Print status update to console every N ticks


# =====================================================
# SECTION 2: IMPORTS AND SETUP
# =====================================================

import os
import random
import math
import multiprocessing
import concurrent.futures
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force headless, non-interactive backend (Must be before pyplot!)
import matplotlib.pyplot as plt

# Apply the global random seed to ensure strict reproducibility 
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Determine available cores for multiprocessing
allocated_cores = multiprocessing.cpu_count() if MAX_WORKERS is None else MAX_WORKERS

print("="*60)
print("HBESD Agent-Based Model Initializing...")
print(f"Targeting {N_AGENTS} agents on a {TOPOLOGY} topology.")
print(f"Computational Cores Allocated: {allocated_cores}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Output Directory: {OUTPUT_DIR}")
print("="*60)

# =====================================================
# SECTION 3: AGENT CLASS DEFINITION
# =====================================================

class Agent:
    """
    The Autopoietic Node. Represents a single human agent governed by the 
    Hybrid Bayesian-Evolutionary Structural Dynamics (HBESD) framework.
    """
    def __init__(self, agent_id, rng):
        self.agent_id = agent_id
        
        # Intrinsic Material Properties (Static)
        self.yield_threshold = rng.uniform(*YIELD_THRESHOLD_RANGE)
        self.parasitic_load = rng.uniform(*PARASITIC_LOAD_RANGE)
        self.integration_efficiency = rng.uniform(*INTEGRATION_EFFICIENCY_RANGE)
        self.structural_entropy = rng.uniform(*STRUCTURAL_ENTROPY_RANGE)
        self.cog_saturation = rng.uniform(*COG_SATURATION_RANGE)
        self.theta_max = THETA_MAX
        
        # Internal State Variables (Dynamic)
        self.prior = rng.uniform(*INITIAL_PRIOR_RANGE)
        self.capacity = rng.uniform(*INITIAL_CAPACITY_RANGE)
        self.agency = rng.uniform(*AGENCY_RANGE)
        self.theta = self.theta_max * self.agency #At t=0, \phi_i(-1) = 0 by convention, so the exhaustion ratio is 1 and \theta_0 = \theta_{\max} \cdot A_i(0).
		
        # Hypothesis Space Variables
        self.h_max = rng.uniform(*H_MAX_RANGE)
        self.h_used = rng.uniform(*H_USED_INIT_RANGE)
        self.lambda_a_max = rng.uniform(*LAMBDA_A_MAX_RANGE)
        self.lambda_a = self.lambda_a_max
        
		# Initialize Hope (rho) based on the feature toggle to guarantee baseline integrity
        if ENABLE_EMERGENT_DYNAMICS:
            self.rho = rng.uniform(*RHO_RANGE)
        else:
            self.rho = RHO  # Locks to the global constant (0.3)
        
        # Kinematic Variables
        self.strain = 0.0
        self.pressure = 0.0
        self.strategy = "homeostatic"
        
        # Channel Topography
        self.channel_weights = {
            "structural": W_STRUCTURAL_INIT,
            "identity": W_IDENTITY_INIT,
            "local": W_LOCAL_INIT
        }
        
        # Failure State tracking
        self.is_ruptured = False
        self.rupture_tick = None
        
        # History Logging
        self.strategy_history = []
        self.capacity_history = []
        self.pressure_history = []
        self.strain_history = []
        self.channel_weight_history = []  # Add this!

    def compute_theta(self, previous_pressure):
        """Calculates metabolic conversion efficiency based on proximity to failure."""
        if self.is_ruptured:
            self.theta = 0.0
            return self.theta
        
        # theta_i(t) = theta_max * A_i(t) * max(0, 1 - phi_i(t-1) / phi*_i)
        exhaustion_ratio = max(0.0, 1.0 - (previous_pressure / self.yield_threshold))
        self.theta = self.theta_max * self.agency * exhaustion_ratio
        return self.theta

    def kl_divergence(self, p_prior, p_post, epsilon=1e-10):
        """Closed-form Bernoulli Kullback-Leibler divergence."""
        # Clamp values to prevent math domain errors (log of zero)
        p_prior = max(epsilon, min(1.0 - epsilon, p_prior))
        p_post = max(epsilon, min(1.0 - epsilon, p_post))
        
        term1 = p_post * math.log(p_post / p_prior)
        term2 = (1.0 - p_post) * math.log((1.0 - p_post) / (1.0 - p_prior))
        return max(0.0, term1 + term2)

    def compute_strain(self, env_signal, anticipated_strain=0.0):
        """Calculates total kinetic and potential strain (S_i)."""
        if self.is_ruptured:
            return 0.0
            
        s_actual = self.kl_divergence(self.prior, env_signal)
        self.strain = max(0.0, s_actual + (ALPHA * anticipated_strain))
        return self.strain

    def compute_pressure(self, anticipated_capacity=0.0):
        """Calculates dimensionless systemic pressure (phi_i)."""
        if self.is_ruptured:
            return float('inf')
        
        # Uses local self.rho (which safely mirrors the global RHO if toggle is False)
        effective_capacity = self.capacity + (self.rho * anticipated_capacity)
        	
        # Guard against zero division - if capacity hits zero, pressure is infinite
        if effective_capacity <= 1e-6:
            self.pressure = float('inf')
        else:
            self.pressure = self.strain / effective_capacity
        
        return self.pressure

    def select_strategy(self):
        """Mechanical selection of the Strategy Triad based on Yield Thresholds."""
        if self.is_ruptured:
            self.strategy = "ruptured"
            return self.strategy

        if self.pressure > self.yield_threshold:
            self.is_ruptured = True
            self.strategy = "plastic_deformation"
            return self.strategy
            
        if self.pressure < (C_TOL * self.yield_threshold):
            self.strategy = "homeostatic"
        elif self.pressure < (C_ACC * self.yield_threshold):
            self.strategy = "acceptance"
        else:
            # Agency and Capacity Gates for Transformation
            if self.agency >= AGENCY_ACTIVATION_THRESHOLD and self.capacity >= CAPACITY_ACTIVATION_FLOOR:
                self.strategy = "transformation"
            else:
                # Default to Removal if Transformation is structurally inaccessible
                # (Network Engine will decide if this is internal or external)
                self.strategy = "removal_pending" 
                
        return self.strategy
    
    def execute_state_updates(self, env_signal, prev_pressure):
        """Applies discrete-time equations to Prior, Capacity, and H-Space."""
        if self.is_ruptured:
            return

        # 1. Strain-Tied Hypothesis Space Dynamics (Option 2)
        if self.strategy == "transformation":
            # Compression scales with strain, bottlenecked by cognitive saturation (omega)
            # Agency acts as the integration coefficient, just as it does for C_i
            compression = H_COMPRESSION_SCALE * self.agency * (self.strain / (self.strain + self.cog_saturation))
            self.h_used = max(0.0, self.h_used - compression)
            
        elif self.strategy == "removal_internal":
            # Epistemic invalidation expands the footprint proportionally to the strain being suppressed
            expansion = H_EXPANSION_SCALE * self.strain
            self.h_used = min(self.h_max, self.h_used + expansion)
            
        elif self.strategy == "homeostatic":
            # Passive decay clears slots exponentially based on structural entropy
            self.h_used = max(0.0, self.h_used * (1.0 - self.structural_entropy))
            
        # "acceptance" and "removal_external" leave h_used structurally unchanged.

        # 2. Calculate dynamic Active Update Rate (lambda_A)
        h_saturation = max(0.0, 1.0 - (self.h_used / self.h_max))
        exhaustion_ratio = max(0.0, 1.0 - (prev_pressure / self.yield_threshold))
        self.lambda_a = self.lambda_a_max * h_saturation * exhaustion_ratio

        # 3. Evaluate Logic Gates based on strategy
        delta_p = 1 if self.strategy in ["acceptance", "transformation"] else 0
        delta_t = 1 if self.strategy == "transformation" else 0

        # 4. Prior Dynamics Update
        if delta_p == 1:
            # Active update uses dynamic lambda_A
            mu = BETA * self.lambda_a
            self.prior = (1.0 - mu) * self.prior + (mu * env_signal)
        else:
            # Passive structural decay uses static structural_entropy (lambda)
            self.prior = (1.0 - self.structural_entropy) * self.prior + (self.structural_entropy * P_BASELINE)
            
        self.prior = max(0.0, min(1.0, self.prior))

        # 5. Capacity Dynamics Update
        passive_drain = self.parasitic_load * self.strain
        
        generation = 0.0
        if delta_t == 1:
            # Hyperbolic capacity generation (Yerkes-Dodson)
            generation = self.integration_efficiency * self.agency * (self.strain / (self.strain + self.cog_saturation))
            
        self.capacity = self.capacity - passive_drain + generation
        
        # Absolute failure floor
        if self.capacity < 1e-6:
            self.capacity = 0.0

    def log_tick(self):
        """Records current state for time-series analysis."""
        self.strategy_history.append(self.strategy)
        self.capacity_history.append(self.capacity)
        self.pressure_history.append(self.pressure)
        self.strain_history.append(self.strain)
        self.channel_weight_history.append(
            self.channel_weights.copy()
        )

# =====================================================
# SECTION 4: NETWORK INITIALIZATION
# =====================================================
# Builds the multiplex grid, instantiates the Agents, and defines the 
# Transmission Propensity matrix (tau_ij).

def build_network(n_agents, topology, p, k, rng):
    """
    Constructs the base network graph.
    """
    # Use the provided random instance to generate a seed for networkx
    nx_seed = rng.randint(0, 2**31 - 1)
    
    if topology == "small_world":
        # k = Each node is joined with its k nearest neighbors in a ring topology.
        # p = The probability of rewiring each edge (creates the "weak ties").
        graph = nx.watts_strogatz_graph(n_agents, k, p, seed=nx_seed)
    elif topology == "random":
        graph = nx.erdos_renyi_graph(n_agents, p, seed=nx_seed)
    else:
        raise ValueError(f"Topology '{topology}' is not supported.")
        
    return graph


def initialize_agents(graph, rng):
    """
    Instantiates an Agent object for every node in the graph.
    Returns a dictionary of {node_id: Agent_object}.
    """
    agent_dict = {}
    for node_id in graph.nodes():
        agent_dict[node_id] = Agent(agent_id=node_id, rng=rng)
    return agent_dict


def initialize_tau_matrix(graph, n_agents, rng):
    """
    Builds the sparse transmission propensity matrix (tau_ij).
    tau_ij represents how easily strain transmits from Node i to Node j.
    """
    # We use LIL (List of Lists) format for efficient matrix construction
    tau_lil = sp.lil_matrix((n_agents, n_agents), dtype=np.float32)
    
    for u, v in graph.edges():
        # Initialize transmission propensity. 
        # For this base implementation, we assign a random starting firewall strength.
        # Edges are bi-directional in the graph, but transmission propensity can be asymmetric.
        tau_lil[u, v] = rng.uniform(0.1, 0.3)
        tau_lil[v, u] = rng.uniform(0.1, 0.3)
        
    # Convert to CSR (Compressed Sparse Row) format for fast math operations later
    tau_csr = tau_lil.tocsr()
    return tau_csr


def compute_sigma(tau_matrix):
    """
    Calculates the Empirical Criticality Parameter (sigma).
    sigma = expected sum of transmission propensities across the local neighborhood.
    Returns a float representing the global thermodynamic phase.
    """
    # Sum the outgoing transmission propensities for each node (sum across columns for each row)
    # tau_matrix.sum(axis=1) returns an (N, 1) matrix of these sums.
    # We then take the mean of these sums to find the network's global sigma.
    node_out_sums = tau_matrix.sum(axis=1)
    sigma_empirical = np.mean(node_out_sums)
    return float(sigma_empirical)

# =====================================================
# SECTION 5: ENVIRONMENTAL ENGINE
# =====================================================
# Manages the external environmental signal (E_k) and macroscopic shocks.

def initialize_environment():
    """
    Sets up the baseline environmental state dictionary.
    """
    return {
        "baseline_signal": P_BASELINE,  # Ambient environmental level (typically neutral 0.5)
        "shock_active": False,
        "shock_remaining_ticks": 0,
        "shock_epicenter_nodes": set()  # Using a set for O(1) fast lookups
    }


def apply_shock(env_state, agent_dict, tick, rng):
    """
    Evaluates the current tick against shock parameters and updates the environment.
    Handles both the initiation of new shocks and the expiration of ongoing ones.
    """
    # 1. Handle ongoing shock duration countdown
    if env_state["shock_active"]:
        env_state["shock_remaining_ticks"] -= 1
        if env_state["shock_remaining_ticks"] <= 0:
            env_state["shock_active"] = False
            env_state["shock_epicenter_nodes"] = set()
            
    # 2. Trigger new shock if we hit the designated tick
    if tick == SHOCK_TICK:
        num_epicenter_nodes = int(N_AGENTS * SHOCK_EPICENTER_PCT)
        node_ids = list(agent_dict.keys())
        
        # Randomly select the nodes that take the direct hit
        hit_nodes = rng.sample(node_ids, num_epicenter_nodes)
        
        env_state["shock_epicenter_nodes"] = set(hit_nodes)
        env_state["shock_active"] = True
        env_state["shock_remaining_ticks"] = SHOCK_DURATION
        
    return env_state


def get_node_signal(node_id, env_state):
    """
    Determines the exact environmental signal (E_k) a specific node observes at this moment.
    """
    if env_state["shock_active"] and node_id in env_state["shock_epicenter_nodes"]:
        return SHOCK_MAGNITUDE
    
    return env_state["baseline_signal"]

# =====================================================
# SECTION 6: TICK EXECUTION ENGINE
# =====================================================
# Orchestrates a single time step of the simulation. Propagates strain across
# the network, updates agent physics, and applies topological decoupling.

def propagate_strain(agent_dict, tau_matrix, anticipated_strains):
    """
    Computes the flow of Bayesian Surprise (Strain) across the network edges.
    Updates the anticipated_strains array in place.
    S_j(t+1) = (1 - ETA) * S_j(t) + sum_i (tau_ij * S_i(t))
    """
    n_agents = len(agent_dict)
    
    # Extract the current kinetic strain (S_i) from all agents into a vector
    current_strains = np.array([agent_dict[i].strain for i in range(n_agents)])
    
    # Calculate total incoming strain from neighbors using sparse dot product
    # tau_matrix[i, j] represents transmission FROM i TO j.
    # Transposing tau (tau.T) and taking the dot product with current_strains yields incoming strain.
    incoming_strain = tau_matrix.T.dot(current_strains)
    
    # Apply attenuation and add incoming
    for i in range(n_agents):
        anticipated_strains[i] = ((1.0 - ETA) * anticipated_strains[i]) + incoming_strain[i]


def execute_tick(tick, agent_dict, tau_matrix, graph, env_state, anticipated_strains, rng):
    """
    The master discrete-time update loop for a single tick.
    """
    # 1. Update the Environment
    env_state = apply_shock(env_state, agent_dict, tick, rng)
    
    # 2. Propagate Strain across the network
    propagate_strain(agent_dict, tau_matrix, anticipated_strains)
    
    # Queue for matrix updates (so we don't modify the matrix while iterating over it)
    tau_updates = []
    
    # 3. Randomize agent execution order to prevent sequential bias
    nodes = list(agent_dict.keys())
    rng.shuffle(nodes)
    
    for node_id in nodes:
        agent = agent_dict[node_id]
        
        if agent.is_ruptured:
            agent.log_tick()
            continue
            
        # Observe the environment
        env_signal = get_node_signal(node_id, env_state)
        
        # --- CORE PHYSICS PIPELINE ---
        prev_pressure = agent.pressure_history[-1] if len(agent.pressure_history) > 0 else 0.0
        agent.compute_theta(prev_pressure)
        agent.compute_strain(env_signal, anticipated_strains[node_id])
        agent.compute_pressure(anticipated_capacity=0.0) # Baseline assumption for now
        
        strategy = agent.select_strategy()
        
        # --- RESOLVE PENDING REMOVAL ---
        if strategy == "removal_pending":
            # Heuristic: 50/50 split between External (cut ties) and Internal (epistemic filter)
            strategy = "removal_external" if rng.random() > 0.5 else "removal_internal"
            agent.strategy = strategy
            
        # --- EXECUTE TOPOLOGICAL & STATE UPDATES ---
        if strategy == "removal_external":
            # Find the neighbor transmitting the highest strain and throttle the connection
            neighbors = list(graph.neighbors(node_id))
            if neighbors:
                worst_neighbor = max(neighbors, key=lambda n: agent_dict[n].strain)
                current_tau = tau_matrix[worst_neighbor, node_id]
                new_tau = max(0.0, current_tau - DELTA_TAU)
                tau_updates.append((worst_neighbor, node_id, new_tau))
            agent.execute_state_updates(env_signal,prev_pressure)
            
        elif strategy == "removal_internal":
            # Collapse attention weighting (Parasitic load of Epistemic Invalidation)
            channels = list(agent.channel_weights.keys())
            target_chan = rng.choice(channels)
            agent.channel_weights[target_chan] = max(0.0, agent.channel_weights[target_chan] - DELTA_W)
            
            # Re-normalize to ensure sum is exactly 1.0
            total_w = sum(agent.channel_weights.values())
            if total_w > 0:
                for c in channels:
                    agent.channel_weights[c] /= total_w
            agent.execute_state_updates(env_signal,prev_pressure)
            
        else:
            # Execute Homeostatic, Acceptance, or Transformation
            agent.execute_state_updates(env_signal,prev_pressure)
            
            # --- EXTENSION: EMERGENT DYNAMICS ---
            # Safely bypassed unless ENABLE_EMERGENT_DYNAMICS is True
            if ENABLE_EMERGENT_DYNAMICS:
                
                # Step 4 & 7: Emergent Altruism (Redistribution)
                if strategy == "homeostatic" and agent.capacity > CAPACITY_SURPLUS_THRESHOLD:
                    neighbors = list(graph.neighbors(node_id))
                    if neighbors:
                        surplus = agent.capacity - CAPACITY_SURPLUS_THRESHOLD
                        transfer_total = surplus * ALTRUISM_TRANSFER_RATE
                        transfer_per_neighbor = transfer_total / len(neighbors)
                        
                        agent.capacity -= transfer_total
                        for n in neighbors:
                            if not agent_dict[n].is_ruptured:
                                agent_dict[n].capacity += transfer_per_neighbor

                # Step 5: Prior Updating Through Observation (Hope Elevation)
                if strategy == "transformation":
                    neighbors = list(graph.neighbors(node_id))
                    for n in neighbors:
                        if not agent_dict[n].is_ruptured:
                            # Boost rho, capping strictly at 1.0 to prevent unmodeled overextension
                            agent_dict[n].rho = min(1.0, agent_dict[n].rho + RHO_OBSERVATION_BOOST)
            
        # --- FINAL YIELD CHECK ---
        # Capacity might have drained to zero during the update, triggering limits
        if agent.capacity <= 1e-6 or agent.pressure > agent.yield_threshold:
            agent.is_ruptured = True
            agent.rupture_tick = tick
            agent.strategy = "plastic_deformation"
            
        agent.log_tick()
        
    # 4. Apply queued topological changes to the Tau Matrix
    if tau_updates:
        tau_lil = tau_matrix.tolil()
        for src, dst, new_val in tau_updates:
            tau_lil[src, dst] = new_val
        tau_matrix = tau_lil.tocsr()
        
    # 5. Generate Tick Summary Statistics
    current_sigma = compute_sigma(tau_matrix)
    ruptured_count = sum(1 for a in agent_dict.values() if a.is_ruptured)
    
    active_capacities = [a.capacity for a in agent_dict.values() if not a.is_ruptured]
    active_pressures = [a.pressure for a in agent_dict.values() if not a.is_ruptured]
    
    mean_cap = np.mean(active_capacities) if active_capacities else 0.0
    mean_pres = np.mean(active_pressures) if active_pressures else 0.0
    
    strat_counts = {
        "homeostatic": 0, "acceptance": 0, "transformation": 0,
        "removal_external": 0, "removal_internal": 0, "plastic_deformation": ruptured_count
    }
    for a in agent_dict.values():
        if not a.is_ruptured and a.strategy in strat_counts:
            strat_counts[a.strategy] += 1

    summary = {
        "tick": tick,
        "sigma": current_sigma,
        "ruptured_count": ruptured_count,
        "mean_capacity": mean_cap,
        "mean_pressure": mean_pres,
        "strategies": strat_counts
    }
    
    return summary, tau_matrix

# =====================================================
# SECTION 7: MAIN SIMULATION LOOP
# =====================================================
# Orchestrates a complete simulation run from initialization to the final tick.

import time

def run_simulation(seed_override=None, override_params=None):
    """
    Executes a full run of the HBESD Agent-Based Model.
    
    :param seed_override: Int to override the RANDOM_SEED (used for batch variance testing).
    :param override_params: Dict of Control Panel variables to temporarily change for this run.
    :return: (results_df, agent_dict, graph, tau_matrix)
    """
    start_time = time.time()
    
    # 1. Apply Temporary Parameter Overrides
    original_globals = {}
    if override_params:
        for key, value in override_params.items():
            if key in globals():
                original_globals[key] = globals()[key]
                globals()[key] = value
            elif not key.startswith('_'):
                print(f"Warning: Override key '{key}' not found in global variables.")

    # 2. Initialize Random Number Generator
    current_seed = seed_override if seed_override is not None else RANDOM_SEED
    rng = random.Random(current_seed)
    
    # 3. Build the World
    graph = build_network(N_AGENTS, TOPOLOGY, CONNECTION_DENSITY, AVG_CONNECTIONS, rng)
    agent_dict = initialize_agents(graph, rng)
    tau_matrix = initialize_tau_matrix(graph, N_AGENTS, rng)
    env_state = initialize_environment()
    
    # Track anticipated strain across the network (initialized to 0)
    anticipated_strains = np.zeros(N_AGENTS)
    
    # 4. Prepare Data Collection
    results = []
    
    print(f"\nStarting Simulation Run (Seed: {current_seed})")
    print("-" * 60)
    
    # 5. Execute Main Time Loop
    for tick in range(TOTAL_TICKS):
        
        # Run the discrete-time update
        tick_summary, tau_matrix = execute_tick(
            tick=tick, 
            agent_dict=agent_dict, 
            tau_matrix=tau_matrix, 
            graph=graph, 
            env_state=env_state, 
            anticipated_strains=anticipated_strains, 
            rng=rng
        )
        
        # Store the results
        results.append(tick_summary)
        
        # Print status update at specific intervals
        if tick % LOG_INTERVAL == 0 or tick == TOTAL_TICKS - 1:
            ruptures = tick_summary["ruptured_count"]
            sig = tick_summary["sigma"]
            press = tick_summary["mean_pressure"]
            strats = tick_summary["strategies"]
            
            print(f"Tick {tick:03d} | Sigma: {sig:>6.4f} | Mean Press: {press:>6.3f} | "
                  f"Ruptures: {ruptures:>4} | H:{strats['homeostatic']:>3} "
                  f"A:{strats['acceptance']:>3} T:{strats['transformation']:>3} "
                  f"R:{strats['removal_external'] + strats['removal_internal']:>3}")

    # 6. Finalize Data
    results_df = pd.DataFrame(results)
    
    # 7. Restore Original Control Panel Parameters
    if override_params:
        for key, value in original_globals.items():
            globals()[key] = value
            
    exec_time = time.time() - start_time
    print("-" * 60)
    print(f"Simulation Complete. Total Exec Time: {exec_time:.2f} seconds.")
    print(f"Final Rupture Count: {results_df.iloc[-1]['ruptured_count']} / {N_AGENTS}")
    
    return results_df, agent_dict, graph, tau_matrix

# =====================================================
# SECTION 8: OUTPUT AND VISUALIZATION
# =====================================================
# Generates the visualizations and CSV exports for the simulation data.

import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force headless, non-interactive backend (Must be before pyplot!)
import matplotlib.pyplot as plt

# Set a clean, academic style for the plots
plt.style.use('ggplot')

def plot_time_series(results_df, output_dir, prefix="baseline"):
    """
    Creates a multi-panel figure showing the network's macro-state over time.
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    ticks = results_df['tick']
    
    # 1. Mean Capacity
    axs[0].plot(ticks, results_df['mean_capacity'], color='blue', linewidth=2)
    axs[0].set_title('Mean Elastic Capacity (C_i) over Time')
    axs[0].set_ylabel('Capacity')
    
    # 2. Mean Pressure
    axs[1].plot(ticks, results_df['mean_pressure'], color='red', linewidth=2)
    axs[1].set_title(r'Mean Systemic Pressure ($\phi_i$) over Time')
    axs[1].set_ylabel(r'Pressure ($\phi_i$)')
    
    # 3. Criticality Parameter (Sigma)
    axs[2].plot(ticks, results_df['sigma'], color='purple', linewidth=2)
    axs[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Criticality Threshold (1.0)')
    axs[2].set_title('Empirical Criticality Parameter (Sigma) over Time')
    axs[2].set_ylabel('Sigma')
    axs[2].legend()
    
    # 4. Strategy Distribution (Stacked Area)
    strats_df = pd.DataFrame(results_df['strategies'].tolist())
    
    if 'removal_external' in strats_df.columns and 'removal_internal' in strats_df.columns:
        strats_df['removal'] = strats_df['removal_external'] + strats_df['removal_internal']
        strats_df = strats_df.drop(columns=['removal_external', 'removal_internal'])
    else:
        strats_df['removal'] = 0
        
    labels = ['homeostatic', 'acceptance', 'transformation', 'removal', 'plastic_deformation']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
    
    for col in labels:
        if col not in strats_df.columns:
            strats_df[col] = 0
            
    axs[3].stackplot(ticks, 
                     strats_df['homeostatic'], 
                     strats_df['acceptance'], 
                     strats_df['transformation'], 
                     strats_df['removal'], 
                     strats_df['plastic_deformation'],
                     labels=['Homeostatic', 'Acceptance', 'Transformation', 'Removal', 'Ruptured'],
                     colors=colors, alpha=0.8)
    
    axs[3].set_title('Strategy Distribution')
    axs[3].set_xlabel('Tick')
    axs[3].set_ylabel('Number of Agents')
    axs[3].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    if SAVE_FIGURES:
        filepath = os.path.join(output_dir, f"{prefix}_timeseries.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[Output] Time series plot saved to {filepath}")
        
    plt.close()


def plot_yerkes_dodson(agent_dict, output_dir, prefix="baseline"):
    """
    Recreates the Yerkes-Dodson curve: Capacity Generation (Delta C_i) vs Shock Magnitude (S_i).
    Plots surviving agents vs. ruptured agents.
    """
    max_strains = []
    delta_capacities = []
    statuses = []
    
    for agent in agent_dict.values():
        if len(agent.strain_history) > 0 and len(agent.capacity_history) > 0:
            max_strain = max(agent.strain_history)
            delta_cap = agent.capacity_history[-1] - agent.capacity_history[0]
            
            max_strains.append(max_strain)
            delta_capacities.append(delta_cap)
            statuses.append('Ruptured' if agent.is_ruptured else 'Survived')
            
    df = pd.DataFrame({
        'Max_Strain': max_strains,
        'Delta_Capacity': delta_capacities,
        'Status': statuses
    })
    
    plt.figure(figsize=(10, 6))
    
    survivors = df[df['Status'] == 'Survived']
    plt.scatter(survivors['Max_Strain'], survivors['Delta_Capacity'], 
                color='blue', alpha=0.6, label='Survived (Active Generation)')
                
    ruptured = df[df['Status'] == 'Ruptured']
    plt.scatter(ruptured['Max_Strain'], ruptured['Delta_Capacity'], 
                color='red', marker='x', alpha=0.4, label='Ruptured (Plastic Deformation)')
                
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Capacity Generation vs. Peak Strain (Yerkes-Dodson Mechanics)')
    plt.xlabel(r'Maximum Bayesian Surprise / Strain ($S_i$)')
    plt.ylabel(r'Change in Elastic Capacity ($\Delta C_i$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if SAVE_FIGURES:
        filepath = os.path.join(output_dir, f"{prefix}_yerkes_dodson.png")
        plt.savefig(filepath, dpi=300)
        print(f"[Output] Yerkes-Dodson plot saved to {filepath}")
        
    plt.close()


def plot_channel_weights(agent_dict, output_dir, prefix="baseline", n_samples=4):
    """
    Plots the Attention Weighting vector (w_i,c) over time for a sample of agents.
    Prioritizes agents that executed Internal Removal to show the channel collapse.
    """
    # Hunt for agents that actually altered their channels
    interesting_agents = [a for a in agent_dict.values() if "removal_internal" in a.strategy_history]
    
    # If none did (e.g., highly stable network), just grab a random sample
    if not interesting_agents:
        interesting_agents = list(agent_dict.values())
        
    sample = random.sample(interesting_agents, min(n_samples, len(interesting_agents)))
    
    fig, axs = plt.subplots(len(sample), 1, figsize=(10, 2.5 * len(sample)), sharex=True)
    if len(sample) == 1:
        axs = [axs]
        
    for i, agent in enumerate(sample):
        history = agent.channel_weight_history
        
        # Unpack the list of dictionaries into separate lines
        str_w = [h["structural"] for h in history]
        id_w = [h["identity"] for h in history]
        loc_w = [h["local"] for h in history]
        ticks = range(len(history))
        
        axs[i].plot(ticks, str_w, label="Structural", color="#2ca02c", linewidth=2)
        axs[i].plot(ticks, id_w, label="Identity", color="#1f77b4", linewidth=2)
        axs[i].plot(ticks, loc_w, label="Local", color="#ff7f0e", linewidth=2)
        
        axs[i].set_title(f"Agent {agent.agent_id} Attention Weighting (Internal Removal executed)")
        axs[i].set_ylabel("Channel Weight")
        axs[i].set_ylim(0, 1.05)
        axs[i].legend(loc="upper right")
        
    plt.xlabel("Tick")
    plt.tight_layout()
    
    if SAVE_FIGURES:
        filepath = os.path.join(output_dir, f"{prefix}_channel_weights.png")
        plt.savefig(filepath, dpi=300)
        print(f"[Output] Channel Weights plot saved to {filepath}")
        
    plt.close()


def save_results_csv(results_df, output_dir, prefix="baseline"):
    """Saves the raw simulation data to CSV."""
    if SAVE_DATA_CSV:
        filepath = os.path.join(output_dir, f"{prefix}_run_summary.csv")
        # Flatten the strategies dict into separate columns for easier parsing
        flat_df = pd.concat([results_df.drop(['strategies'], axis=1), 
                             results_df['strategies'].apply(pd.Series)], axis=1)
        flat_df.to_csv(filepath, index=False)
        print(f"[Output] Data CSV saved to {filepath}")


def generate_all_outputs(results_df, agent_dict, output_dir, prefix="baseline"):
    """Wrapper function to generate all requested outputs."""
    print("\n" + "-"*60)
    print("GENERATING OUTPUT FILES...")
    print("-" * 60)
    plot_time_series(results_df, output_dir, prefix)
    plot_yerkes_dodson(agent_dict, output_dir, prefix)
    plot_channel_weights(agent_dict, output_dir, prefix)
    save_results_csv(results_df, output_dir, prefix)
    print("-" * 60 + "\n")

# =====================================================
# SECTION 9: BATCH RUNNER & PARAMETER SWEEPS
# =====================================================
# Automates parallel execution of multiple simulations to map the phase space
# and generate the core falsifiable predictions (Figures 2, 3, and 4).

import itertools
import os
import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use('Agg')  # Force headless, non-interactive backend (Must be before pyplot!)
import matplotlib.pyplot as plt

def _parallel_worker(override_dict):
    """
    Helper function for the ProcessPoolExecutor. 
    Runs a single simulation and extracts just the final macro-state data to save memory.
    """
    df, agents, _, _ = run_simulation(seed_override=RANDOM_SEED, override_params=override_dict)
    
    # We only need the final tick's summary for the heatmaps
    final_state = df.iloc[-1]
    rupture_rate = final_state['ruptured_count'] / N_AGENTS
    
    # For the Yerkes-Dodson dense scatter, we need agent-level data
    agent_data = []
    if override_dict.get("_extract_agent_data", False):
        import random
        # MEMORY PROTECTION: Dynamically sample agents if N is very large
        # Target ~1000 agents per shock magnitude for a clean, readable scatter plot
        sample_fraction = min(1.0, 1000 / N_AGENTS)
        
        for a in agents.values():
            if len(a.strain_history) > 0 and len(a.capacity_history) > 0:
                if random.random() <= sample_fraction:
                    agent_data.append({
                        "max_strain": max(a.strain_history),
                        "delta_c": a.capacity_history[-1] - a.capacity_history[0],
                        "ruptured": a.is_ruptured
                    })
                
    return {
        "params": override_dict,
        "rupture_rate": rupture_rate,
        "agent_data": agent_data
    }

def run_agency_gamma_sweep():
    """Generates Data for Figure 2: The Event Horizon."""
    print("\nStarting Agency vs. Parasitic Load Sweep (Figure 2)...")
    
    a_vals = np.arange(0.1, 1.1, 0.1)
    g_vals = np.arange(0.01, 0.16, 0.02)
    
    tasks = []
    for a, g in itertools.product(a_vals, g_vals):
        tasks.append({
            "AGENCY_RANGE": (a, min(1.0, a + 0.05)),
            "PARASITIC_LOAD_RANGE": (g, g + 0.01),
            "_a_val": a,  # Store for plotting
            "_g_val": g
        })
        
    results_matrix = np.zeros((len(g_vals), len(a_vals)))
    
    with concurrent.futures.ProcessPoolExecutor( max_workers=allocated_cores ) as executor:
        results = list(executor.map(_parallel_worker, tasks))
        
    for res in results:
        a_idx = np.where(np.isclose(a_vals, res["params"]["_a_val"]))[0][0]
        g_idx = np.where(np.isclose(g_vals, res["params"]["_g_val"]))[0][0]
        results_matrix[g_idx, a_idx] = res["rupture_rate"]
        
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(results_matrix, origin='lower', aspect='auto', cmap='inferno',
               extent=[a_vals[0], a_vals[-1], g_vals[0], g_vals[-1]])
    plt.colorbar(label='Rupture Rate (1.0 = Total Collapse)')
    plt.axvline(x=AGENCY_ACTIVATION_THRESHOLD, color='white', linestyle='--', linewidth=2, label='Transformation Gate (A*)')
    plt.title('Systemic Event Horizon: Agency vs. Parasitic Load')
    plt.xlabel(r'Baseline Agency ($A_i$)')
    plt.ylabel(r'Parasitic Load ($\gamma$)')
    plt.legend()
    plt.grid(False)
    
    filepath = os.path.join(OUTPUT_DIR, "fig2_agency_gamma_heatmap.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"-> Figure 2 saved to {filepath}")


def run_eustress_distress_sweep():
    """Generates Data for Figure 3: The Dense Yerkes-Dodson Curve."""
    print("\nStarting Eustress/Distress Sweep (Figure 3)...")
    
    # Sweep shock magnitudes from 0.05 up to 0.999
    shock_vals = np.concatenate((np.arange(0.05, 0.95, 0.05), [0.999]))
    
    tasks = []
    for s in shock_vals:
        tasks.append({
            "SHOCK_MAGNITUDE": float(s),
            "_extract_agent_data": True # Tell worker to return node-level data
        })
        
    all_agents = []
    with concurrent.futures.ProcessPoolExecutor( max_workers=allocated_cores ) as executor:
        results = list(executor.map(_parallel_worker, tasks))
        for res in results:
            all_agents.extend(res["agent_data"])
            
    # Plotting
    max_strains_surv = [a["max_strain"] for a in all_agents if not a["ruptured"]]
    delta_c_surv = [a["delta_c"] for a in all_agents if not a["ruptured"]]
    
    max_strains_rupt = [a["max_strain"] for a in all_agents if a["ruptured"]]
    delta_c_rupt = [a["delta_c"] for a in all_agents if a["ruptured"]]

    plt.figure(figsize=(10, 6))
    plt.scatter(max_strains_surv, delta_c_surv, color='blue', alpha=0.3, s=10, label='Survived (Active Generation)')
    plt.scatter(max_strains_rupt, delta_c_rupt, color='red', marker='x', alpha=0.3, s=10, label='Ruptured (Phase Transition Graveyard)')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title('Capacity Generation vs. Peak Strain (Multi-Run Aggregate)')
    plt.xlabel(r'Maximum Bayesian Surprise / Strain ($S_i$)')
    plt.ylabel(r'Change in Elastic Capacity ($\Delta C_i$)')
    plt.legend()
    
    filepath = os.path.join(OUTPUT_DIR, "fig3_eustress_distress.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"-> Figure 3 saved to {filepath}")


def run_stress_inoculation_sweep():
    """Generates Data for Figure 4: Shock Timing vs Initial Prior."""
    print("\nStarting Stress Inoculation Sweep (Figure 4)...")
    
    # Use exact ranges from your LaTeX document
    prior_ranges = [(0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.4, 0.6)]
    shock_ticks = range(0, 31, 2)
    
    tasks = []
    for p_range, t in itertools.product(prior_ranges, shock_ticks):
        tasks.append({
            "INITIAL_PRIOR_RANGE": p_range,
            "SHOCK_TICK": t,
            "_p_mid": sum(p_range)/2.0,
            "_t_val": t
        })
        
    results_matrix = np.zeros((len(prior_ranges), len(shock_ticks)))
    p_mids = [sum(pr)/2.0 for pr in prior_ranges]
    
    with concurrent.futures.ProcessPoolExecutor( max_workers=allocated_cores ) as executor:
        results = list(executor.map(_parallel_worker, tasks))
        
    for res in results:
        p_idx = np.where(np.isclose(p_mids, res["params"]["_p_mid"]))[0][0]
        t_idx = list(shock_ticks).index(res["params"]["_t_val"])
        results_matrix[p_idx, t_idx] = res["rupture_rate"]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(results_matrix, origin='lower', aspect='auto', cmap='magma',
               extent=[shock_ticks[0], shock_ticks[-1], p_mids[0], p_mids[-1]])
    plt.colorbar(label='Rupture Rate')
    plt.title('Stress Inoculation: Shock Timing vs. Initial Prior')
    plt.xlabel('Shock Arrival (Tick)')
    plt.ylabel('Initial Prior Expectation (Midpoint)')
    plt.grid(False)
    
    filepath = os.path.join(OUTPUT_DIR, "fig4_stress_inoculation_heatmap.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"-> Figure 4 saved to {filepath}")

# =====================================================
# SECTION 10: VALIDATION CHECKS (ISOLATED UNIT TESTS)
# =====================================================
# These functions mathematically verify that the Agent class strictly obeys 
# the structural equations of the HBESD framework.

def validate_kl_properties(dummy_agent, n_samples=1000):
    """
    Verifies that the closed-form Bernoulli Kullback-Leibler divergence 
    behaves correctly: D_KL >= 0, D_KL(P||P) == 0, and asymmetry.
    """
    failures = []
    
    # 1. Self-Divergence should be zero
    p_test = 0.5
    if dummy_agent.kl_divergence(p_test, p_test) != 0.0:
        failures.append("FAIL: KL(P||P) is not exactly 0.0")

    # 2. Non-negativity and Asymmetry
    asymmetry_confirmed = False
    for _ in range(n_samples):
        p = random.uniform(0.01, 0.99)
        q = random.uniform(0.01, 0.99)
        
        kl_pq = dummy_agent.kl_divergence(p, q)
        kl_qp = dummy_agent.kl_divergence(q, p)
        
        if kl_pq < 0 or kl_qp < 0:
            failures.append(f"FAIL: Negative KL divergence found for P={p}, Q={q}")
            
        # Due to floating point math, we look for a meaningful difference to confirm asymmetry
        if abs(kl_pq - kl_qp) > 1e-5:
            asymmetry_confirmed = True

    if not asymmetry_confirmed:
        failures.append("FAIL: KL Divergence failed to demonstrate asymmetry.")
        
    return "PASS" if not failures else "\n".join(failures)


def validate_phi_floor(dummy_agent):
    """
    Verifies the critical limit equation: lim_{C_i -> 0} phi_i = infinity.
    A node with zero capacity must experience infinite systemic pressure.
    """
    # Force strain to be positive
    dummy_agent.strain = 1.5 
    
    # Force effective capacity to zero
    dummy_agent.capacity = 0.0 
    
    pressure = dummy_agent.compute_pressure(anticipated_capacity=0.0)
    
    if pressure == float('inf'):
        return "PASS"
    else:
        return f"FAIL: Zero capacity returned pressure of {pressure} instead of infinity."


def validate_prior_bounds(dummy_agent):
    """
    Verifies the convex combination equation keeps the Prior strictly within [0,1].
    """
    # Force extreme values to test the bounds
    dummy_agent.prior = 0.99
    dummy_agent.strategy = "acceptance" # Forces delta_P = 1
    
    # Hit it with a massive shock
    dummy_agent.execute_state_updates(env_signal=1.0)
    
    if 0.0 <= dummy_agent.prior <= 1.0:
        return "PASS"
    else:
        return f"FAIL: Prior breached [0,1] bounds. Current value: {dummy_agent.prior}"


def validate_channel_weights(dummy_agent):
    """
    Verifies the zero-sum nature of the Attention Weighting vector (sum w_ic = 1.0).
    """
    total_weight = sum(dummy_agent.channel_weights.values())
    
    # Using math.isclose to handle minor floating point inaccuracies
    if math.isclose(total_weight, 1.0, rel_tol=1e-9):
        return "PASS"
    else:
        return f"FAIL: Channel weights sum to {total_weight}, expected 1.0"


def run_agent_unit_tests():
    """Executes all isolated physics checks on a dummy agent."""
    print("\n" + "-"*50)
    print("EXECUTING MATHEMATICAL VALIDATION CHECKS (PHASE 1)")
    print("-"*50)
    
    # Initialize a dummy agent using the global RNG seed
    dummy_agent = Agent(agent_id=9999, rng=random)
    
    print(f"[Check] KL Divergence Properties... {validate_kl_properties(dummy_agent)}")
    print(f"[Check] Yield Threshold (Phi) Floor...  {validate_phi_floor(dummy_agent)}")
    print(f"[Check] Prior Structural Bounds...      {validate_prior_bounds(dummy_agent)}")
    print(f"[Check] Channel Weight Integrity...     {validate_channel_weights(dummy_agent)}")
    print("-"*50 + "\n")


# =====================================================
# MASTER EXECUTION BLOCK
# =====================================================
if __name__ == "__main__":
    import sys
    
    # Simple CLI router
    print("\n" + "="*60)
    print("HBESD SIMULATION ENGINE")
    print("="*60)
    print("1. Run Baseline Simulation (Outputs single-run charts & CSV)")
    print("2. Run Parameter Sweeps (Generates Figures 2, 3, and 4)")
    print("3. Run ALL")
    
    choice = input("\nSelect execution mode (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        print("\nExecuting Baseline Simulation...")
        df_results, final_agents, final_graph, final_tau = run_simulation()
        generate_all_outputs(df_results, final_agents, OUTPUT_DIR, prefix="baseline")
        
    if choice in ['2', '3']:
        print("\nExecuting Multiprocessed Parameter Sweeps...")
        print(f"Utilizing {allocated_cores} CPU cores. This may take a few minutes.")
        run_agency_gamma_sweep()
        run_eustress_distress_sweep()
        run_stress_inoculation_sweep()
        
    print("\nAll requested processes complete. Check the ./hbesd_output directory.")