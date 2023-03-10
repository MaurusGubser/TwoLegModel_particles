import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os


def compute_hist_stats(X_hist):
    hist_mean = np.mean(X_hist, axis=1)
    hist_var = np.var(X_hist, axis=1)
    return [hist_mean, hist_var]


class Plotter:
    def __init__(self, true_states, true_obs, delta_t):
        nb_steps_states, _, dim_states = true_states.shape
        nb_steps_obs, _, dim_observations = true_obs.shape
        if nb_steps_states != nb_steps_obs:
            raise AssertionError(
                "States and observation have different number of steps: {} and {}, respectively".format(nb_steps_states,
                                                                                                        nb_steps_obs))
        self.nb_steps = nb_steps_states
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.true_states = true_states
        self.true_obs = true_obs
        self.contact_left = np.zeros((nb_steps_obs, 1))
        self.contact_right = np.zeros((nb_steps_obs, 1))
        self.set_contacts()
        self.delta_t = delta_t
        self.t_vals = np.arange(0, self.nb_steps)
        self.state_names = [
                "$x_H$",
                "$y_H$",
                r"$\varphi_0$",
                r"$\varphi_1$",
                r"$\varphi_2$",
                r"$\varphi_3$",
                r"$\dot x_H$",
                r"$\dot y_H$",
                r"$\dot \varphi_0$",
                r"$\dot \varphi_1$",
                r"$\dot \varphi_2$",
                r"$\dot \varphi_3$",
                r"$\ddot x_H$",
                r"$\ddot y_H$",
                r"$\ddot \varphi_0$",
                r"$\ddot \varphi_1$",
                r"$\ddot \varphi_2$",
                r"$\ddot \varphi_3$",
            ]

    def set_contacts(self):
        self.contact_left = self.true_obs[:, :, 13]
        self.contact_right = self.true_obs[:, :, 17]

    def plot_particles_trajectories(self, X_hist):
        nb_steps, nb_particles, dim_states = X_hist.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                "Truth and particles are not compatible: shape truth is {}; shape particles is {}".format(
                    self.true_states.shape, X_hist.shape))

        nb_trajectories = min(nb_particles, 5)
        nb_axes = 3
        nb_figures = int(np.ceil(self.dim_states / nb_axes))
        if self.dim_states == 18:
            state_names = [
                "$x_H$",
                "$y_H$",
                r"$\varphi_0$",
                r"$\varphi_1$",
                r"$\varphi_2$",
                r"$\varphi_3$",
                r"$\dot x_H$",
                r"$\dot y_H$",
                r"$\dot \varphi_0$",
                r"$\dot \varphi_1$",
                r"$\dot \varphi_2$",
                r"$\dot \varphi_3$",
                r"$\ddot x_H$",
                r"$\ddot y_H$",
                r"$\ddot \varphi_0$",
                r"$\ddot \varphi_1$",
                r"$\ddot \varphi_2$",
                r"$\ddot \varphi_3$",
            ]
        else:
            raise AssertionError("Dimension of state vector expected to be 18; got {}".format(self.dim_states))

        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(9, 6))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis="both")
                for k in range(0, nb_trajectories):
                    axs[j].plot(self.t_vals, X_hist[:, k, nb_axes * i + j], lw=1)
                axs[j].plot(self.t_vals, self.true_states[:, :, nb_axes * i + j], label="truth", lw=1.5, color="green")
                axs[j].set_title(state_names[nb_axes * i + j])
                if state_names[nb_axes * i + j] == r"$y_H$":
                    axs[j].plot(self.t_vals, self.contact_left + 1.0, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right + 1.0, label="Contact right", color="orange", lw=1.5)
                elif state_names[nb_axes * i + j] == r"$\dot x_H$":
                    axs[j].plot(self.t_vals, self.contact_left + 0.6, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right + 0.6, label="Contact right", color="orange", lw=1.5)
                else:
                    axs[j].plot(self.t_vals, self.contact_left, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right, label="Contact right", color="orange", lw=1.5)
                axs[j].legend()
            fig.suptitle("{} particle trajectories".format(nb_trajectories))
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            plt.show()
        return None

    def plot_smoothed_trajectories(self, samples):
        nb_steps, nb_samples, dim_states = samples.shape
        assert nb_steps == self.nb_steps and dim_states == self.dim_states, \
            "Truth and states are not compatible: shape truth is {}; shape samples is {}".format(self.true_states.shape,
                                                                                                 samples.shape)
        nb_trajectories = min(nb_samples, 5)
        nb_axes = 3
        nb_figures = int(np.ceil(self.dim_states / nb_axes))
        if self.dim_states == 18:
            state_names = [
                "$x_H$",
                "$y_H$",
                r"$\varphi_0$",
                r"$\varphi_1$",
                r"$\varphi_2$",
                r"$\varphi_3$",
                r"$\dot x_H$",
                r"$\dot y_H$",
                r"$\dot \varphi_0$",
                r"$\dot \varphi_1$",
                r"$\dot \varphi_2$",
                r"$\dot \varphi_3$",
                r"$\ddot x_H$",
                r"$\ddot y_H$",
                r"$\ddot \varphi_0$",
                r"$\ddot \varphi_1$",
                r"$\ddot \varphi_2$",
                r"$\ddot \varphi_3$",
            ]
        else:
            raise AssertionError("Dimension of state vector expected to be 18; got {}".format(self.dim_states))

        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(9, 6))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis="both")
                for k in range(0, nb_trajectories):
                    axs[j % nb_axes].plot(self.t_vals, samples[:, k, nb_axes * i + j], lw=1)
                axs[j % nb_axes].plot(self.t_vals, self.true_states[:, :, nb_axes * i + j], label="truth", lw=1.5,
                                      color="green")
                if state_names[nb_axes * i + j] == r"$y_H$":
                    axs[j].plot(self.t_vals, self.contact_left + 1.0, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right + 1.0, label="Contact right", color="orange", lw=1.5)
                elif state_names[nb_axes * i + j] == r"$\dot x_H$":
                    axs[j].plot(self.t_vals, self.contact_left + 0.6, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right + 0.6, label="Contact right", color="orange", lw=1.5)
                else:
                    axs[j].plot(self.t_vals, self.contact_left, label="Contact left", color="red", lw=1.5)
                    axs[j].plot(self.t_vals, self.contact_right, label="Contact right", color="orange", lw=1.5)
                axs[j].set_title(state_names[nb_axes * i + j])
                axs[j].legend()
            fig.suptitle("Smoothed samples")
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            plt.show()
        return None


def plot_particle_moments(self, particles_mean, particles_var, name_suffix=""):
    nb_axes = 3
    nb_figures = int(np.ceil(self.dim_states / nb_axes))
    particles_sd = np.sqrt(particles_var)
    if self.dim_states == 18:
        state_names = [
            "$x_H$",
            "$y_H$",
            r"$\varphi_0$",
            r"$\varphi_1$",
            r"$\varphi_2$",
            r"$\varphi_3$",
            r"$\dot x_H$",
            r"$\dot y_H$",
            r"$\dot \varphi_0$",
            r"$\dot \varphi_1$",
            r"$\dot \varphi_2$",
            r"$\dot \varphi_3$",
            r"$\ddot x_H$",
            r"$\ddot y_H$",
            r"$\ddot \varphi_0$",
            r"$\ddot \varphi_1$",
            r"$\ddot \varphi_2$",
            r"$\ddot \varphi_3$",
        ]
    else:
        raise AssertionError("Dimension of state vector expected to be 18; got {}".format(self.dim_states))
    fig_list = []
    axs_list = []
    for i in range(0, nb_figures):
        fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(9, 6))
        for j in range(0, nb_axes):
            if i * nb_axes + j > self.dim_states - 1:
                break
            axs[j].grid(axis="both")
            axs[j].plot(self.t_vals, particles_mean[:, nb_axes * i + j], label="Particle mean", color="blue")
            axs[j].fill_between(self.t_vals, particles_mean[:, nb_axes * i + j] - particles_sd[:, nb_axes * i + j],
                                particles_mean[:, nb_axes * i + j] + particles_sd[:, nb_axes * i + j], alpha=0.2,
                                color="blue")
            axs[j].plot(self.t_vals, self.true_states[:, :, nb_axes * i + j], label="Truth", color="green", lw=1.5)
            if state_names[nb_axes * i + j] == r"$y_H$":
                axs[j].plot(self.t_vals, self.contact_left + 1.0, label="Contact left", color="red", lw=1.5)
                axs[j].plot(self.t_vals, self.contact_right + 1.0, label="Contact right", color="orange", lw=1.5)
            elif state_names[nb_axes * i + j] == r"$\dot x_H$":
                axs[j].plot(self.t_vals, self.contact_left + 0.6, label="Contact left", color="red", lw=1.5)
                axs[j].plot(self.t_vals, self.contact_right + 0.6, label="Contact right", color="orange", lw=1.5)
            else:
                axs[j].plot(self.t_vals, self.contact_left, label="Contact left", color="red", lw=1.5)
                axs[j].plot(self.t_vals, self.contact_right, label="Contact right", color="orange", lw=1.5)
            axs[j].set_title(state_names[nb_axes * i + j])
            axs[j].legend()

        fig.suptitle("Particles mean and variance")
        fig.tight_layout()
        fig_list.append(fig)
        axs_list.append(axs)
        plt.show()
    return None


def plot_observations(self, samples, observation_map):
    nb_steps, nb_samples, dim_states = samples.shape
    nb_trajectories = min(nb_samples, 5)
    if nb_steps != self.nb_steps or dim_states != self.dim_states:
        raise AssertionError(
            "Truth and states are not compatible: shape truth is {}; shape samples is {}".format(self.true_states.shape,
                                                                                                 samples.shape))
    true_obs = self.true_obs
    obs = np.empty((nb_steps, nb_trajectories, self.dim_observations))
    for i in range(0, nb_trajectories):
        obs[:, i, :] = observation_map(samples[:, i, :])

    if self.dim_observations == 20:
        obs_names = [
            "$\ddot x^0$",
            "$\ddot y^0$",
            "$\omega_z^0$",
            "$\ddot x^1$",
            "$\ddot y^1$",
            "$\omega_z^1$",
            "$\ddot x^2$",
            "$\ddot y^2$",
            "$\omega_z^2$",
            "$\ddot x^3$",
            "$\ddot y^3$",
            "$\omega_z^3$",
            "$\dot x^4$",
            "$\dot y^4$",
            "$\ddot x^4$",
            "$\ddot y^4$",
            "$\dot x^5$",
            "$\dot y^5$",
            "$\ddot x^5$",
            "$\ddot y^5$",
        ]
    else:
        raise ValueError("Observation dimension has to be 20; got {} instead.".format(self.dim_observations))
    _, _, nb_observations = obs.shape
    nb_axes = 3
    nb_figures = int(np.ceil(nb_observations / nb_axes))
    for i in range(0, nb_figures):
        fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(9, 6))
        for j in range(0, nb_axes):
            if i * nb_axes + j > nb_observations - 1:
                break
            axs[j].grid(axis="both")
            for k in range(0, nb_trajectories):
                axs[j].plot(self.t_vals, obs[:, k, i * nb_axes + j], lw=1)
            axs[j].plot(self.t_vals, true_obs[:, :, i * nb_axes + j], label="True observation", color="green", lw=1.5)
            if obs_names[i * nb_axes + j] in ["$\ddot y^0$", "$\ddot y^1$", "$\ddot y^2$", "$\ddot y^3$", ]:
                axs[j].plot(self.t_vals, self.contact_left + 10.0, label="Contact left", color="red", lw=1.5)
                axs[j].plot(self.t_vals, self.contact_right + 10.0, label="Contact right", color="orange", lw=1.5)
            else:
                axs[j].plot(self.t_vals, self.contact_left, label="Contact left", color="red", lw=1.5)
                axs[j].plot(self.t_vals, self.contact_right, label="Contact right", color="orange", lw=1.5)
            axs[j].legend()
            axs[j].set_title(obs_names[i * nb_axes + j])
        fig.tight_layout()
        plt.show()
    return None


def plot_ESS(self, ESS):
    window_avg = np.ones(10) / 10.0
    moving_avg = np.convolve(ESS, window_avg, "same")
    fig = plt.figure(figsize=(9, 6))
    plt.grid(axis="both")
    plt.plot(self.t_vals, ESS, label="ESS")
    plt.plot(self.t_vals, moving_avg, label="ESS Moving Avg")
    plt.plot(self.t_vals, self.contact_left, label="Contact left", color="red", lw=1.5)
    plt.plot(self.t_vals, self.contact_right, label="Contact right", color="orange", lw=1.5)
    fig.suptitle("Essential sample size")
    plt.legend()
    plt.show()
    return None


def plot_logLts_one_run(self, logLts):
    fig = plt.figure(figsize=(9, 6))
    plt.grid(axis="both")
    plt.plot(self.t_vals, logLts)
    plt.xlabel("t")
    plt.ylabel(r"$\log(p(y_{0:t}))$")
    fig.suptitle("Log likelihood")
    plt.show()
    return None


def compute_residuals(self, observations):
    residuals = np.abs(self.true_obs - observations)
    return residuals


def plot_logLts_multiple_runs(self, output_multismc, nb_particles, nb_runs, t_trunc):
    logLts_data = {}
    for N in nb_particles:
        logLts_data[N] = np.array([r["output"].summaries.logLts for r in output_multismc if r["N"] == N])

    # temporal progress of logLts
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axs[0].grid(axis="both")
    axs[1].grid(axis="both")
    for key, loglts in logLts_data.items():
        mean = np.mean(loglts, axis=0)
        std = np.std(loglts, axis=0)
        axs[0].plot(self.t_vals, mean, label="N={}".format(key))
        # axs[0].fill_between(self.t_vals, mean - std, mean + std, alpha=0.5)
        axs[0].legend()
        axs[0].set_xlabel("t")
        axs[0].set_ylabel(r"$\log(p(y_{0:t}))$")
        axs[0].set_title("Mean with std deviation")
        axs[1].plot(self.t_vals, std ** 2, label="N={}".format(key))
        axs[1].legend()
        axs[1].set_xlabel("t")
        axs[1].set_ylabel(r"Variance of $\log(p(y_{0:t}))$")
        axs[1].set_title(r"Variance of $\log(p(y_{0:t}))$")
        fig.suptitle("Non-truncated loglikelihood, averaged over {} runs".format(nb_runs))

    # comparison of differently truncated logLts
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axs[0].grid(axis="both")
    axs[1].grid(axis="both")
    t_start_vals = np.arange(0, t_trunc + 1, t_trunc // 50)
    for key, loglts in logLts_data.items():
        logLts_truncated = np.array([loglts[:, -1] - loglts[:, t] for t in t_start_vals]).T
        mean_truncated = np.mean(logLts_truncated, axis=0)
        std_trunacted = np.std(logLts_truncated, axis=0)
        axs[0].plot(t_start_vals, mean_truncated, label="N={}".format(key))
        axs[0].set_xlabel("Start time $ t_{0} $")
        axs[0].set_ylabel(r"$\log(p(y_{t_{0}+1:T} | y_{0:t_{0}})$")
        axs[0].set_title("Mean with std deviation")
        axs[0].legend()
        axs[1].plot(t_start_vals, std_trunacted ** 2, label="N={}".format(key))
        axs[1].set_xlabel("Start time $ t_{0} $")
        axs[1].set_title(r"Variance of $ \log(p(y_{t_{0}+1:T} | y_{0:t_{0}})) $")
        axs[1].legend()
    fig.suptitle("Truncated loglikelihood for different starting times, averaged over {} runs".format(nb_runs))

    # nb_particles-vs-var plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axs[0].grid(axis="both")
    axs[1].grid(axis="both")
    means_N = np.array([np.mean(loglts[:, -1]) for loglts in logLts_data.values()])
    axs[0].plot(nb_particles, means_N, marker="o")
    axs[0].set_xlabel("N")
    axs[0].set_ylabel(r"$\log(p(y_{0:T}))$")
    axs[0].set_title("Loglikelihood")
    means_N_truncated = np.array([np.mean(loglts[:, -1] - loglts[:, t_trunc]) for loglts in logLts_data.values()])
    axs[1].plot(nb_particles, means_N_truncated, marker="o")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel(r"$\log(p(y_{t_{0}:T} | y_{0:t_{0}+1}))$")
    axs[1].set_title("Truncated loglikelihood")
    fig.suptitle("N vs. Loglikelihood, truncated at {}, averaged over {} runs".format(t_trunc, nb_runs))

    # boxplot of truncated logLts
    fig = plt.figure(figsize=(12, 8))
    sb.boxplot(x=[r["output"].summaries.logLts[-1] - r["output"].summaries.logLts[t_trunc] for r in output_multismc],
               y=[str(r["N"]) for r in output_multismc], showfliers=False)
    plt.xlabel("$\log(p(y_{t_{0}+1:T} | y_{0:t_{0}}))$")
    plt.ylabel("Number of particles")
    fig.suptitle("Loglikelihood truncated at {}, averaged over {} runs".format(t_trunc, nb_runs))

    # hists of truncated logLts
    fig = plt.figure(figsize=(12, 8))
    sb.histplot(x=[r["output"].summaries.logLts[-1] - r["output"].summaries.logLts[t_trunc] for r in output_multismc],
                hue=[str(r["N"]) for r in output_multismc], multiple="dodge")
    plt.xlabel(r"Bins of $\log(p(y_{t_{0}+1:T}|y_{0:t_{0}}))$")
    fig.suptitle("Histogram of loglikelihood truncated at {}, averaged over {} runs ".format(t_trunc, nb_runs))
    plt.show()
    return None


def plot_likelihood_parameters(self, output_multismc, model_params, t_trunc):
    logLts = [r["output"].summaries.logLts[-1] for r in output_multismc]
    plt.figure(figsize=(12, 8))
    sb.boxplot(x=logLts, y=[r["fk"] for r in output_multismc], showfliers=False)
    plt.title("Boxplots for loglikelihood, non-truncated")
    logLts_truncated = [r["output"].summaries.logLts[-1] - r["output"].summaries.logLts[t_trunc] for r in
                        output_multismc]
    plt.figure(figsize=(12, 8))
    sb.boxplot(x=logLts_truncated, y=[r["fk"] for r in output_multismc], showfliers=False)
    plt.title("Boxplots for loglikelihood, truncated at {}".format(t_trunc))
    plt.figure(figsize=(12, 8))
    for fk_model in model_params:
        logLts = np.array([r["output"].summaries.logLts for r in output_multismc if r["fk"] == fk_model])
        mean, std = np.mean(logLts, axis=0), np.std(logLts, axis=0)
        print("Parameters={}; mean of loglikelihood={};\nmean of truncated likelihood={}".format(fk_model, mean[-1],
                                                                                                 mean[-1] - mean[
                                                                                                     t_trunc]))
        plt.plot(self.t_vals, mean, label=fk_model)
        plt.xlabel("Timesteps")
        plt.ylabel("$p(y_{0:t})$")
        plt.legend()
        plt.title("Non-truncated loglikelihood")
    plt.show()
    return None


def plot_learned_parameters(self, alg, learning_alg, prior_dict):
    if learning_alg == "pmmh" or learning_alg == "cpmmh" or learning_alg == "gibbs":
        burnin = 0  # discard the __ first iterations
        for i, param in enumerate(prior_dict.keys()):
            plt.figure(figsize=(9, 6))
            sb.histplot(alg.chain.theta[param][burnin:])
            plt.title(param)
        plt.show()
    elif learning_alg == "smc2":
        for i, param in enumerate(prior_dict.keys()):
            plt.figure(figsize=(9, 6))
            sb.histplot([t[i] for t in alg.X.theta])
            plt.title(param)
        plt.show()
    else:
        raise ValueError("learning_alg has to be one of 'pmmh', 'gibbs', 'smc2'; got {} instead.".format(learning_alg))
    return None
