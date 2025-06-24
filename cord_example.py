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
    return (mo,)


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
