using Random
using POMDPs, StatsBase, POMDPToolbox, RLInterface, Parameters, GridInterpolations
using AutomotiveDrivingModels, AutoViz
using POMDPModelTools, POMDPPolicies, POMDPSimulators, BeliefUpdaters
using Reel
using QMDP, SARSOP
using AutomotivePOMDPs

include("policy_plot.jl")

rng = MersenneTwister(1);

params = CrosswalkParams()
#obstacle_1 = ConvexPolygon([VecE2(5, -2),    VecE2(5, -5),
#                            VecE2(20, -5),   VecE2(20, -2)],4)
#obstacle_2 = ConvexPolygon([VecE2(5, +4.5),  VecE2(5, +7.5),
#                            VecE2(22, +7.5), VecE2(22, +4.5)],4)
#params.obstacles = [obstacle_1, obstacle_2]
params.obstacles_visible = true

env   = CrosswalkEnv(params)
pomdp = SingleOCPOMDP(env=env, ΔT = 0.5, p_birth = 0.5, no_ped_prob=0.0)

# solve random policy
random_solver = RandomSolver(rng)
random_policy = solve(random_solver, pomdp)

# solve QMDP policy
qmdp_solver   = QMDPSolver(max_iterations=100, belres=1e-5, verbose=true)
qmdp_policy   = solve(qmdp_solver, pomdp)

# solve SARSOP policy
sarsop_solver = SARSOPSolver(fast=true, randomization=true, precision = 0.05, timeout = 1000.)
sarsop_policy = solve(sarsop_solver, pomdp)

policy_plot(pomdp, qmdp_policy,   "plots/qmdp_0.00001.pdf", n_pts=300, n_bins=40)
policy_plot(pomdp, sarsop_policy, "plots/sarsop_0.05.pdf",  n_pts=300, n_bins=40)

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
