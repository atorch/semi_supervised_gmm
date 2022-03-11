import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal


def update_params_hat(df, pr_y_given_x):

    n_classes = pr_y_given_x.shape[1]

    params_hat = {
        "n_classes": n_classes,
        "pr_y": pr_y_given_x.mean(axis=0),
        "mu_x": [],
        "vcov_x": [],
    }

    for c in range(n_classes):

        params_hat["mu_x"].append(
            np.average(df[["x1", "x2"]], weights=pr_y_given_x[:, c], axis=0)
        )

        # TODO Make sure this is correct, check ddof
        params_hat["vcov_x"].append(
            np.cov(df[["x1", "x2"]], rowvar=False, aweights=pr_y_given_x[:, c])
        )

    return params_hat


def calculate_pr_y_given_x(df, params_hat):

    n_obs = df.shape[0]

    # Could also get this from len(params_hat["pr_y"])
    n_classes = params_hat["n_classes"]

    multivariate_normals = [
        multivariate_normal(mean=params_hat["mu_x"][c], cov=params_hat["vcov_x"][c])
        for c in range(n_classes)
    ]

    pr_y_given_x = np.zeros((n_obs, n_classes))
    for i in range(n_obs):
        observed_class = df["observed_y"].iloc[i]

        if not np.isnan(observed_class):

            # We are absolutely certain of Y for this observation (it is observed)
            pr_y_given_x[i, int(observed_class)] = 1.0

        else:

            # Y is not observed, we need to calculate Pr[ Y | X ]
            observed_x = df[["x1", "x2"]].iloc[i]
            priors_times_densities = np.zeros((n_classes,))
            for c in range(n_classes):
                priors_times_densities[c] = params_hat["pr_y"][
                    c
                ] * multivariate_normals[c].pdf(observed_x)

            pr_y_given_x[i, :] = priors_times_densities / np.sum(priors_times_densities)

    # Sanity check
    assert np.isclose(pr_y_given_x.sum(), n_obs)

    return pr_y_given_x


def run_expectation_maximization(df, n_classes, max_iter=20):

    # TODO Randomize initial values of mu_x?
    # Try multiple random starting values, return best likelihood
    # Note that [0, 0] and n=2 both hardcode the dimension of the predictors ["x1", "x2"]
    params_hat = {
        "pr_y": [
            1 / n_classes,
        ]
        * n_classes,
        "mu_x": [
            [0, 0],
        ]
        * n_classes,
        "vcov_x": [
            np.identity(n=2),
        ]
        * n_classes,
        "n_classes": n_classes,
    }

    for iter_idx in range(max_iter):
        print(f"EM iteration {iter_idx}")

        ## E step: update pr_y_given_x
        pr_y_given_x = calculate_pr_y_given_x(df, params_hat)

        ## M step: update params_hat
        params_hat = update_params_hat(df, pr_y_given_x)

    return params_hat


def simulate_x(df, mu_x, vcov_x, n_classes):

    # These will be overwritten with real values
    df["x1"] = 0
    df["x2"] = 0

    for c in range(n_classes):

        idx_class = np.where(df["true_y"] == c)[0].tolist()
        n_obs = len(idx_class)

        # We simulate from Pr[ X | Y == c]
        X = np.random.multivariate_normal(mean=mu_x[c], cov=vcov_x[c], size=n_obs)
        df.loc[idx_class, ["x1", "x2"]] = X

    return df


def simulate(
    n_obs,
    pr_y,
    mu_x,
    vcov_x,
    pr_y_is_observed,
):

    n_classes = len(pr_y)

    df = pd.DataFrame(
        {"true_y": np.random.choice(range(n_classes), size=n_obs, p=pr_y)}
    )

    df = simulate_x(df, mu_x, vcov_x, n_classes)

    df["observed_y"] = df["true_y"]

    n_not_observed = int((1 - pr_y_is_observed) * n_obs)
    idx_not_observed = np.random.choice(
        range(n_obs), size=n_not_observed, replace=False
    )

    # We use NaNs to represent Ys that are not observed
    df.loc[idx_not_observed, "observed_y"] = np.nan

    return df


def main():

    # TODO Set np random seed

    # Entry i is the mean of X conditional on Y==i
    # The predictors X are in R^2
    mu_x = [
        [4.5, 0.5],
        [-1, -3],
        [1, 1],
        [5, 4],
    ]

    # Entry i is the variance-covariance matrix of X conditional on Y==i
    # Make sure these are positive (semi-) definite
    vcov_x = [
        [
            [2.0, -0.3],
            [-0.3, 1.0],
        ],
        [
            [2, 0],
            [0, 3],
        ],
        [
            [2, 1],
            [1, 4],
        ],
        [
            [3.0, 1.5],
            [1.5, 2.0],
        ],
    ]

    pr_y = [0.2, 0.3, 0.4, 0.1]
    df_train = simulate(
        n_obs=500,
        pr_y=pr_y,
        mu_x=mu_x,
        vcov_x=vcov_x,
        pr_y_is_observed=0.05,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    x1 = np.array(df_train["x1"])
    x2 = np.array(df_train["x2"])
    for c in range(len(pr_y)):
        idx_class = np.where(df_train["true_y"] == c)[0].tolist()
        plt.plot(
            x1[idx_class], x2[idx_class], label=f"Y = {c}", marker="o", linestyle=""
        )

    ax.legend()
    plt.savefig("plot_simulation_x_and_true_y.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(len(pr_y)):
        idx_class = np.where(df_train["observed_y"] == c)[0].tolist()
        plt.plot(
            x1[idx_class], x2[idx_class], label=f"Y = {c}", marker="o", linestyle=""
        )

    idx_y_not_observed = np.where(df_train["observed_y"].isnull())[0].tolist()
    plt.plot(
        x1[idx_y_not_observed],
        x2[idx_y_not_observed],
        label=f"Y is not observed",
        marker="x",
        linestyle="",
        alpha=0.5,
    )
    ax.legend()
    plt.savefig("plot_simulation_x_and_observed_y.png")
    plt.clf()

    # EM is not allowed to use true_y
    # It only sees observed_y (which contains many NaNs)
    params_hat = run_expectation_maximization(
        df=df_train[["x1", "x2", "observed_y"]], n_classes=len(pr_y), max_iter=30
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    # TODO Could be fun to have a similar plot showing path of EM params_hat over time (iteration by iteration)
    for c in range(params_hat["n_classes"]):
        color = next(ax._get_lines.prop_cycler)["color"]
        plt.plot(
            mu_x[c][0],
            mu_x[c][1],
            marker="s",
            linestyle="",
            label=f"true E[ X | Y = {c}]",
            color=color,
        )
        plt.plot(
            params_hat["mu_x"][c][0],
            params_hat["mu_x"][c][1],
            marker="+",
            linestyle="",
            label=f"estimated E[ X | Y = {c}]",
            color=color,
        )

    ax.legend()
    plt.savefig("plot_mu_x_true_and_estimated.png")
    plt.clf()

    # TODO Also plot Pr[ Y | X ], or plot gaussian ellipses around E[ X | Y ] to visualize estimated vcov matrices

    # TODO Evaluate model on a test set


if __name__ == "__main__":
    main()
