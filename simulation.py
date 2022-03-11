import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    # Entry i is the mean of X conditional on Y==i
    # The predictors X are in R^2
    mu_x = [
        [2.5, 1],
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


if __name__ == "__main__":
    main()
