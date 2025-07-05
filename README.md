# bayes-cord

The Critical Odds-Ratio Discrimination (CORD) test is an idea for a Bayesian hypothesis testing methodology loosely inspired by the Region Of Practical Equivalence ([ROPE](https://easystats.github.io/bayestestR/articles/region_of_practical_equivalence.html)).

The test, given a contingency table with positive and negative counts for both "control" and "exposed" populations, computes the posterior distribution on two possible parameters $p_c$ and $p_e$. Then, considering the odds ratio: 

$$
r = \frac{p_e(1-p_c)}{(1-p_e)p_c}
$$

it takes a log odds-ratio width $w$, finds an interval $[e^{-w}, e^w]$, and determines what is the probability that odds ratio is above, below, or inside this band.

This repository provides an implementation of the test as well as a few examples of using it with synthetic data.

## How to use

### Virtual environment and package management

The repository uses [uv](https://docs.astral.sh/uv/) for package management. Check out [the official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once it's set up on your system, you can use it to create the virtual environment simply with:

```bash
uv sync
```

in the root of the repository.

### Example

There's an example using a [marimo](https://marimo.io/) notebook. Run it by using:

```bash
uv run marimo run cord_example.py
```

Or edit it with:

```bash
uv run marimo edit
```

This should open the notebook interface in your browser.

### How to use

The test is implemented as the function `cord_test` in the `bayes_cord.py` module. The docstring describes its API:

```py
def cord_test(table: NDArray[np.int_], w: float = 0.0) -> CordTestResult:
    """
    Perform a Critical Odds-Ratio Discrimination (CORD) test on a 2x2 
    contingency table.
    
    Args:
        table [NDArray[np.int_]]:
            A 2x2 numpy array representing the contingency table.
            The first column corresponds to the amounts of positive and negative
            results for the condition of interest with no exposure,
            and the second column corresponds to the amounts of positive and
            negative results for the condition of interest with exposure.
        w [float]:
            A float representing the width of log-odds ratio central interval.
            Default is 0.0, which means no interval is considered.
    
    Returns:
        CordTestResult:
            A dataclass containing the odds ratios and the upper, middle, and 
            lower bands of the CORD test.
    """
```

The result type contains the members:

* `or_up`: upper odds ratio bound
* `or_down`: lower odds ratio bound
* `upper_band`: probability of the odds ratio being in the upper band ($r \ge r_{up}$)
* `middle_band`: probability of the odds ratio being the middle band ($r_{up} > r > r_{down}$)
* `lower_band`: probability of the odds ratio being the lower band ($r \le r_{down}$)