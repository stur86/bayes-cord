import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from bayes_cord import cord_test
    from test_data import SyntheticCTable
    import matplotlib.pyplot as plt
    return SyntheticCTable, cord_test, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The Critical Odds-Ratio Discrimination (CORD) test is an idea for a Bayesian hypothesis testing methodology loosely inspired by the Region Of Practical Equivalence ([ROPE](https://easystats.github.io/bayestestR/articles/region_of_practical_equivalence.html)).

    The test, given a contingency table with positive and negative counts for both "control" and "exposed" populations, computes the posterior distribution on two possible parameters $p_c$ and $p_e$. Then, considering the odds ratio: 

    $$
    r = \frac{p_e(1-p_c)}{(1-p_e)p_c}
    $$

    it takes a log odds-ratio width $w$, finds an interval $[e^{-w}, e^w]$, and determines what is the probability that odds ratio is above, below, or inside this band.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The central region is defined by the following rule:

    $$
    p_e^{low} = e^{-w}\frac{p_c}{1+(e^{-w}-1)p_c}
    $$

    $$
    p_e^{up} = e^w\frac{p_c}{1+(e^w-1)p_c}
    $$

    $$
    p_e^{low} < p_e < p_e^{up}
    $$


    To integrate the probability on the lower band we find:

    $$
    B_{low} = \int_0^1\int_0^{p_e^{low}} p_c^{a}(1-p_c)^bp_e^c(1-p_e)^d \frac{\Gamma(a+b)\Gamma(c+d)}{\Gamma(a)\Gamma(b)\Gamma(c)\Gamma(d)} dp_c dp_e = \int_0^1 p_c^{a}(1-p_c)^b I\left(e^{-w}\frac{p_c}{1+(e^{-w}-1)p_c};c, d\right)
    \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}dp_c
    $$

    for a contingency table with elements $a, b, c, d$. Here $I(x;a, b)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function). A similar integral holds for the upper region, and the middle region can be simply found by subtracting the other two from 1. While the integral is in theory a polynomial, for anything but very small counts it is computationally intractable due to high oscillations and powers, and we use numerical integration instead.
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    def plot_beta_function(ctable: np.ndarray, ax: plt.Axes, n: int = 50):
        grid = np.linspace(0, 1, n)
        x, y = np.meshgrid(grid, grid)
        ctable = np.asarray(ctable)
        a, b = ctable[:,0]
        c, d = ctable[:,1]

        z = x**a*(1-x)**b*y**c*(1-y)**d

        ax.set_aspect(1.0)
        ax.set_xlabel("$p_1$")
        ax.set_ylabel("$p_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.pcolormesh(x, y, z)

    def plot_beta_bound(oddr: float, ax: plt.Axes, n: int = 100, c: str = 'w'):
        x = np.linspace(0, 1, n)
        y = oddr*x/(1+(oddr-1)*x)
        ax.plot(x, y, c=c)

    def plot_beta_area(oddr1: float, oddr2: float, ax: plt.Axes, n: int = 100, c: str = 'w', alpha: float =0.3):
        x = np.linspace(0, 1, n)
        y1 = oddr1*x/(1+(oddr1-1)*x)
        if oddr2 == np.inf:
            y2 = np.ones_like(x)
        elif oddr2 == -np.inf:
            y2 = np.zeros_like(x)
        else:
            y2 = oddr2*x/(1+(oddr2-1)*x)
        ax.fill_between(x, y1, y2, color=c, alpha=alpha)
    return plot_beta_area, plot_beta_bound, plot_beta_function


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here is an example of CORD applied to a test contingency table:

    |                  | Did not study | Did study |
    |------------------|--------------:|----------:|
    | **Did pass**     |              5|         12|
    | **Did not pass** |              6|          4|

    The question being asked is whether studying improves or not the chances of passing the test, and how probable it is to do so. $p_1$ will be the probability of passing without studying, and $p_2$ that of passing after studying.
    """
    )
    return


@app.cell
def _(cord_test, np, plot_beta_area, plot_beta_bound, plot_beta_function, plt):

    # Try multiplying this table by an integer factor to see how a larger sample size with the
    # same relative frequencies results in higher confidence
    _test_ctable = np.array([[5, 12], [6, 4]])
    _log_or_w = 0.6

    _cord_res = cord_test(_test_ctable, _log_or_w)
    _or_up = _cord_res.or_up
    _or_down = _cord_res.or_down

    _fig, _ax = plt.subplots(1, 3, figsize=(14,4.5))

    _fig.suptitle(f"CORD areas, log odds ratio width: {_log_or_w}")

    # Colors
    _c_up = '#fa5555'
    _c_mid = '#55fa55'
    _c_down = '#5555fa'

    _ax[0].set_title(f"P: {_cord_res.upper_band:.3f}")
    plot_beta_function(_test_ctable, _ax[0])
    plot_beta_bound(_or_up, _ax[0], c=_c_up)
    plot_beta_area(_or_up, np.inf, _ax[0], c=_c_up)

    _ax[1].set_title(f"P: {_cord_res.middle_band:.3f}")
    plot_beta_function(_test_ctable, _ax[1])
    plot_beta_bound(_or_up, _ax[1], c=_c_mid)
    plot_beta_bound(_or_down, _ax[1], c=_c_mid)
    plot_beta_area(_or_up, _or_down, _ax[1], c=_c_mid)

    _ax[2].set_title(f"P: {_cord_res.lower_band:.3f}")
    plot_beta_function(_test_ctable, _ax[2])
    plot_beta_bound(_or_down, _ax[2], c=_c_down)
    plot_beta_area(_or_down, -np.inf, _ax[2], c=_c_down)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now let's see another example, this time using synthetic data:""")
    return


@app.cell
def _(SyntheticCTable, cord_test, mo):
    # Try now with synthetic data simulating a very rare event
    _true_r = 0.5
    _synth_data = SyntheticCTable(p1=0.02, odds_ratio=_true_r, seed=0)
    _ctable = _synth_data.generate(1000)

    _cord_res = cord_test(_ctable, 0.25)

    mo.md(f"""
    |     |  $\\neg X$ | $X$ |
    |-----|-----------:|----:|
    | $Y$ | {_ctable[0,0]} | {_ctable[0,1]} |
    | $\\neg Y$ | {_ctable[1,0]} | {_ctable[1,1]} | 

    True odds ratio: $r = {_true_r}$.

    $P(r \\ge {_cord_res.or_up:.3f}) = {_cord_res.upper_band:.3f}$  
    $P({_cord_res.or_up:.3f} > r > {_cord_res.or_down:.3f}) = {_cord_res.middle_band:.3f}$  
    $P(r \\le {_cord_res.or_down:.3f}) = {_cord_res.lower_band:.3f}$ 

    ```
    {_cord_res}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
