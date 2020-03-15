using Random
using POMDPs, StatsBase, POMDPToolbox, RLInterface, Parameters, GridInterpolations
using AutomotiveDrivingModels, AutoViz
using POMDPModelTools, POMDPPolicies, POMDPSimulators, BeliefUpdaters
using Reel
using QMDP, SARSOP
using AutomotivePOMDPs
using ProgressMeter
using Statistics

include("policy_plot.jl")

rng = MersenneTwister(1);

# general crosswalk setup
params = CrosswalkParams(obstacles_visible=true, n_max_ped=1)
env    = CrosswalkEnv(params)

sarsop_pomdp  = SingleOCPOMDP(env=env, collision_cost=-30., pos_obs_noise=0.5, vel_obs_noise=0.5)
sarsop_solver = SARSOPSolver(precision=0.05, timeout=1000.)
sarsop_policy = solve(sarsop_solver, sarsop_pomdp)
policy_plot(sarsop_pomdp, sarsop_policy, "sarsopSIM_0_01_1000_noise_05", n_pts=300, n_bins=40, sig=0.5)

qmdp_pomdp  = SingleOCPOMDP(env=env, collision_cost=-1.6, pos_obs_noise=0.5, vel_obs_noise=0.5)
qmdp_solver = QMDPSolver(max_iterations=100, belres=1e-5, verbose=true)
qmdp_policy = solve(qmdp_solver, qmdp_pomdp)
policy_plot(qmdp_pomdp, qmdp_policy, "qmdpSIM_0_00001_noise_05", n_pts=300, n_bins=40, sig=0.5)

## optimal study
"""
n_episodes = 10000

# compute random policy
random_solver = RandomSolver(rng)
random_policy = solve(random_solver, SingleOCPOMDP(env=env, collision_cost=-1., pos_obs_noise=0.5, vel_obs_noise=0.5))

# simulate random policy
random_pomdp_sim = SingleOCPOMDP(env=env, ΔT=0.1, p_birth=0.01, collision_cost=-1., pos_obs_noise=0.5,  vel_obs_noise=0.5)
random_updater   = NothingUpdater()

time_to_cross = Set()
collision     = zeros(1, n_episodes)

@showprogress for i in 1:n_episodes

    hr = HistoryRecorder(rng=rng, max_steps=300)
    hist = POMDPs.simulate(hr, random_pomdp_sim, random_policy, random_updater);

    collision[i] = hist.hist[end].s.crash
    collision[i] == 0 ? push!(time_to_cross, hist.hist[end].t*random_pomdp_sim.ΔT) : nothing

end

@show collision_rate_mean = mean(collision*100)
@show collision_rate_std  = std(collision*100)
@show time_to_cross_mean  = mean(time_to_cross)
@show time_to_cross_std   = std(time_to_cross)

n_episodes = 100

# compute qmdp policy
qmdp_pomdp  = SingleOCPOMDP(env=env, collision_cost=-1.6, pos_obs_noise=0.5, vel_obs_noise=0.5)
qmdp_solver = QMDPSolver(max_iterations=100, belres=1e-5, verbose=true)
qmdp_policy = solve(qmdp_solver, qmdp_pomdp)
policy_plot(qmdp_pomdp, qmdp_policy, "qmdpSIM_0_00001_noise_05", n_pts=300, n_bins=40, sig=0.5)

# simulate qmdp policy
qmdp_pomdp_sim = SingleOCPOMDP(env=env, ΔT=0.1, p_birth=0.01, collision_cost=-1.6, pos_obs_noise=0.5,  vel_obs_noise=0.5)
qmdp_updater   = SingleOCUpdater(qmdp_pomdp_sim);
hr = HistoryRecorder(rng=rng, max_steps=300)

time_to_cross = Set()
collision     = zeros(1, n_episodes)

@showprogress for i in 1:n_episodes

    hist = POMDPs.simulate(hr, qmdp_pomdp_sim, qmdp_policy, qmdp_updater);

    collision[i] = hist.hist[end].s.crash
    collision[i] == 0 ? push!(time_to_cross, hist.hist[end].t*qmdp_pomdp_sim.ΔT) : nothing

end

@show collision_rate_mean = mean(collision)
@show collision_rate_std  = std(collision)
@show time_to_cross_mean  = mean(time_to_cross)
@show time_to_cross_std   = std(time_to_cross)


# compute sarsop policy
sarsop_pomdp  = SingleOCPOMDP(env=env, collision_cost=-30., pos_obs_noise=0.5, vel_obs_noise=0.5)
sarsop_solver = SARSOPSolver(precision=0.05, timeout=1000.)
sarsop_policy = solve(sarsop_solver, sarsop_pomdp)
policy_plot(sarsop_pomdp, sarsop_policy, "sarsopSIM_0_01_1000_noise_05", n_pts=300, n_bins=40, sig=0.5)

# simulate sarsop policy
sarsop_pomdp_sim = SingleOCPOMDP(env=env, ΔT=0.1, p_birth=0.01, collision_cost=-30., pos_obs_noise=0.5,  vel_obs_noise=0.5)
sarsop_updater = SingleOCUpdater(sarsop_pomdp_sim);
hr = HistoryRecorder(rng=rng, max_steps=300)

time_to_cross = Set()
collision     = zeros(1, n_episodes)

@showprogress for i in range(1, n_episodes)

    hist = POMDPs.simulate(hr, sarsop_pomdp_sim, sarsop_policy, sarsop_updater);

    collision[i] = hist.hist[end].s.crash
    collision[i] == 0 ? push!(time_to_cross, hist.hist[end].t*sarsop_pomdp_sim.ΔT) : nothing

end

@show collision_rate_mean = mean(collision)
@show collision_rate_std  = std(collision)
@show time_to_cross_mean  = mean(time_to_cross)
@show time_to_cross_std   = std(time_to_cross)


## pomdp setup for sensor uncertainty study

noise_00_pomdp = SingleOCPOMDP(env=env, collision_cost=-10., pos_obs_noise=0.01, vel_obs_noise=0.01)
noise_05_pomdp = SingleOCPOMDP(env=env, collision_cost=-10., pos_obs_noise=0.5,  vel_obs_noise=0.5)
noise_10_pomdp = SingleOCPOMDP(env=env, collision_cost=-10., pos_obs_noise=1.0,  vel_obs_noise=1.0)
noise_20_pomdp = SingleOCPOMDP(env=env, collision_cost=-10., pos_obs_noise=2.0,  vel_obs_noise=2.0)



# solve SARSOP policy
@showprogress for pomdp in [noise_00_pomdp, noise_05_pomdp, noise_10_pomdp, noise_20_pomdp]

    sarsop_solver = SARSOPSolver(randomization=true, precision = 0.01, timeout = 1000.)
    sarsop_policy = solve(sarsop_solver, pomdp)

    σ = pomdp.pos_obs_noise
    #policy_plot1(pomdp, sarsop_policy, "sarsop_0.05_600_noise_$(σ)_1", n_pts=300, n_bins=40, sig=σ)
    #policy_plot2(pomdp, sarsop_policy, "sarsop_0.05_600_noise_$(σ)_2", n_pts=300, n_bins=40, sig=σ)
    policy_plot3(pomdp, sarsop_policy, "sarsop_0.01_1000_noise_$(σ)", n_pts=300, n_bins=40, sig=σ)

end


# solve QMDP policy
@showprogress for pomdp in [noise_00_pomdp, noise_05_pomdp, noise_10_pomdp, noise_20_pomdp]

    qmdp_solver   = QMDPSolver(max_iterations=100, belres=1e-5, verbose=true)
    qmdp_policy   = solve(qmdp_solver, pomdp)

    σ = pomdp.pos_obs_noise
    policy_plot(pomdp, qmdp_policy, "qmdp_0.00001_noise_$σ", n_pts=300, n_bins=40, sig=σ)

end
"""


"""
up     = updater(policy);

hr = HistoryRecorder(rng=rng, max_steps=200)
@time hist = POMDPs.simulate(hr, pomdp, policy, up);

# duration, fps, render_hist = animate_hist(pomdp, hist, SceneOverlay[])
# film = roll(render_hist, fps = fps, duration = duration)
# write("singleOCPOMDP_random.gif", film);

up = SingleOCUpdater(pomdp);

hr = HistoryRecorder(rng=rng)
@time hist = POMDPs.simulate(hr, pomdp, qmdp_policy, up);

duration, fps, render_hist = animate_hist(pomdp, hist)
film = roll(render_hist, fps = fps, duration = duration)
write("singleOCPOMDP_qmdp.gif", film);

pomdp.p_birth=0.5

hr = HistoryRecorder(rng=rng)
@time hist = POMDPs.simulate(hr, pomdp, sarsop_policy, DiscreteUpdater(pomdp));

duration, fps, render_hist = animate_hist(pomdp, hist)
film = roll(render_hist, fps = fps, duration = duration)
write("singleOCPOMDP_sarsop.gif", film);
"""
