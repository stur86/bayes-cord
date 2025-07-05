import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.special import betainc, betaincc, beta
from scipy.integrate import quad

@dataclass
class CordTestResult:
    """Dataclass to hold the results of the CORD test.
    
    Attributes:
        or_up: float
            The upper odds ratio.
        or_down: float
            The lower odds ratio.
        upper_band: float
            The upper band of the CORD test.
        middle_band: float
            The middle band of the CORD test.
        lower_band: float
            The lower band of the CORD test.
    """
    or_up: float
    or_down: float
    upper_band: float
    middle_band: float
    lower_band: float

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
    
    # We consider p1 the probability of x11 and p2 the probability of x12
    a, b = table[:,0]+1
    c, d = table[:,1]+1
    
    norm_ab = beta(a, b)    
    
    or_up = np.exp(w)
    or_down = np.exp(-w)
        
    def kernel_up(x: NDArray[np.number]) -> NDArray[np.number]:
        y = or_up*x/(1+(or_up-1)*x)
        return betaincc(c, d, y)*x**(a-1)*(1.0-x)**(b-1)/norm_ab
    
    def kernel_down(x: NDArray[np.number]) -> NDArray[np.number]:
        y = or_down*x/(1+(or_down-1)*x)
        return betainc(c, d, y)*x**(a-1)*(1.0-x)**(b-1)/norm_ab
    
    
    band_down, _ = quad(kernel_down, 0, 1)
    band_up, _ = quad(kernel_up, 0, 1)
    
    band_middle = 1.0 - band_down - band_up
    
    return CordTestResult(
        or_up=or_up,
        or_down=or_down,
        upper_band=band_up,
        middle_band=band_middle,
        lower_band=band_down
    )
    
