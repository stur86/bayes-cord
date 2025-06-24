import numpy as np
from numpy.typing import NDArray


class SyntheticCTable:
    """
    A class to represent a synthetic 2x2 contingency table for testing purposes.
    
    Attributes:
        table (np.ndarray):
            A 2x2 numpy array representing the contingency table.
    """
    
    def __init__(self, p1: float, odds_ratio: float, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._p1 = p1
        self._odds_ratio = odds_ratio
        self._p2 = p1 * odds_ratio / (1 + p1 * (odds_ratio - 1))
        
    def generate(self, count: int = 100, p_x: float = 0.5) -> NDArray[np.int32]:
        """
        Generate a synthetic 2x2 contingency table.
        
        Args:
            count (int): The number of samples to generate for each cell.
            p_x (float): Probability of exposure in the population.
        
        Returns:
            np.ndarray: A 2x2 numpy array representing the contingency table.
        """        
        
        # Exposure
        c1 = self._rng.binomial(count, p_x)
        c2 = count - c1
        
        x11 = self._rng.binomial(c1, self._p1)
        x12 = self._rng.binomial(c2, self._p2)
        x21 = c1 - x11
        x22 = c2 - x12
                
        return np.array([[x11, x12], [x21, x22]], dtype=np.int32)
    
    def __repr__(self) -> str:
        return f"SyntheticCTable(p1={self._p1}, p2={self._p2}, odds_ratio={self._odds_ratio}"
    
if __name__ == "__main__":
    # Example usage
    synthetic_table = SyntheticCTable(p1=0.01, odds_ratio=2.0, seed=42)
    print(synthetic_table)
    table = synthetic_table.generate(count=1000, p_x=0.5)
    print(table)