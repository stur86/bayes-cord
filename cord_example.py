import marimo

__generated_with = "0.14.9"
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

    band_colors = [
        '#fa5555',
        '#55fa55',
        '#5555fa'
    ]
    return band_colors, plot_beta_area, plot_beta_bound, plot_beta_function


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
def _(
    band_colors,
    cord_test,
    np,
    plot_beta_area,
    plot_beta_bound,
    plot_beta_function,
    plt,
):

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
    _c_up = band_colors[0]
    _c_mid = band_colors[1]
    _c_down = band_colors[2]

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
    log_or_slider = mo.ui.slider(start=0.0, stop=4.0, step=0.1)
    return (log_or_slider,)


@app.cell(hide_code=True)
def _(log_or_slider, mo, np, plot_beta_bound, plot_beta_function, plt):
    _fig, _ax = plt.subplots(figsize=(5,5))

    _test_ctable = np.array([[5, 12], [6, 4]])
    plot_beta_function(_test_ctable, _ax)
    plot_beta_bound(np.exp(log_or_slider.value), _ax)
    plot_beta_bound(np.exp(-log_or_slider.value), _ax)

    mo.vstack([
        mo.md("""
        Here we can set our log odds width and see the corresponding interval plotted.
        """),
        mo.hstack(["Log odds ratio width: ", log_or_slider, f"{log_or_slider.value}"], justify="start"),
        mo.left(_fig)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now let's see another example, this time using synthetic data. We simulate a case control study. For example, imagine we're trying to detect whether exposure to a certain chemical increases the likelihood of a certain rare disease. The baseline rate for this disease is 2%, and it may be increased by an unknown factor (our odds ratio) upon exposure. Probability of exposure is 20%.

    In these conditions, it would be hard to get a representative sample by random sampling. Instead, we sample 50 known people with the disease and verify their exposure status; then we sample the same number of people without the disease and do the same. We then compute our three bands based on these numbers.
    """
    )
    return


@app.cell
def _(SyntheticCTable, band_colors, cord_test, mo, plt):
    # Try now with synthetic data simulating a very rare event
    _baseline_rate = 0.02
    _true_r = 1.5
    _p_exposure = 0.2
    _synth_data = SyntheticCTable(p1=_baseline_rate, odds_ratio=_true_r, seed=0)
    _ctable = _synth_data.generate(100, p_x=0.2, case_control=True)

    _cord_res = cord_test(_ctable, 0.25)

    _fig, _ax = plt.subplots()
    _ax.set_title(f"Odds ratio middle band: {_cord_res.or_down:.3f} - {_cord_res.or_up:.3f}")

    _ax.set_ylim(0, 1)
    _ax.bar(["Lower band", "Middle band", "Upper band"], [_cord_res.lower_band, _cord_res.middle_band, _cord_res.upper_band], 
        color=band_colors)
    _ax.text(0, _cord_res.lower_band+0.02, f"{_cord_res.lower_band:.1%}", size='large', weight='bold', ha='center')
    _ax.text(1, _cord_res.middle_band+0.02, f"{_cord_res.middle_band:.1%}", size='large', weight='bold', ha='center')
    _ax.text(2, _cord_res.upper_band+0.02, f"{_cord_res.upper_band:.1%}", size='large', weight='bold', ha='center')

    mo.vstack([
        mo.md(f"""
    |     |  $\\neg X$ | $X$ |
    |-----|-----------:|----:|
    | $Y$ | {_ctable[0,0]} | {_ctable[0,1]} |
    | $\\neg Y$ | {_ctable[1,0]} | {_ctable[1,1]} | 

    Baseline rate: $p_c = {_baseline_rate*100:.1f}\\%$  
    True odds ratio: $r = {_true_r}$  
    Exposure probability: $p_x = {_p_exposure*100:.1f}\\%$

    $P(r \\ge {_cord_res.or_up:.3f}) = {_cord_res.upper_band:.3f}$  
    $P({_cord_res.or_up:.3f} > r > {_cord_res.or_down:.3f}) = {_cord_res.middle_band:.3f}$  
    $P(r \\le {_cord_res.or_down:.3f}) = {_cord_res.lower_band:.3f}$ 

    ```
    {_cord_res}
    ```
    """), mo.left(_fig)])
    return


@app.cell
def _(SyntheticCTable, band_colors, cord_test, mo, np, plt):
    _baseline_rate = 0.1
    _odds_ratio_range = np.linspace(1.5, 4, 10)
    _seed = 0
    _tests = 5
    _samples = 200

    _band_points = []

    for _or in _odds_ratio_range:
        _sdata = SyntheticCTable(p1=_baseline_rate, odds_ratio=_or, seed=_seed)
        _band_points.append([])
        for _ in range(_tests):
            _ctable_rnd = _sdata.generate(_samples, p_x=0.5)
            _ctable_cc = _sdata.generate(_samples, p_x=0.5, case_control=True)

            _res_rnd = cord_test(_ctable_rnd, np.log(1.25))
            _res_cc = cord_test(_ctable_cc, np.log(1.25))
            _band_points[-1].append([_res_rnd.lower_band, _res_rnd.middle_band, _res_rnd.upper_band,
                                     _res_cc.lower_band, _res_cc.middle_band, _res_cc.upper_band])
        _seed += 1

    _or_axis = np.repeat(_odds_ratio_range, _tests)
    _band_points = np.array(_band_points).reshape((-1,6))*100

    _fig, _ax = plt.subplots()
    _ax.set_title(f"Band probabilities at different odds ratios, {_samples} samples")

    _ax.plot([_or_axis[0], _or_axis[-1]], [50, 50], c='w', lw=0.5, ls='--')
    _ax.set_xlim(_or_axis[0], _or_axis[-1])
    for _b_i in range(3):
        _bname = ["lower", "middle", "upper"]
        _ax.scatter(_or_axis, _band_points[:,_b_i], c=band_colors[_b_i], marker='x', lw=0.5, label=f"{_bname[_b_i]} random")
        _ax.scatter(_or_axis, _band_points[:,_b_i+3], c=band_colors[_b_i], marker='^', lw=0.5, alpha=0.5, label=f"{_bname[_b_i]} case control")
    _ax.set_xlabel("True odds ratio")
    _ax.set_ylabel("P(r > 1.25)")

    _ax.legend()

    mo.vstack([
        mo.md(f"""
        Here is an example of how well the test does at detecting that the odds ratio is in the upper band for different samples, in both randomized and case control studies, with a baseline rate of {_baseline_rate:.1%}.
    """),
        mo.left(_fig)
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
