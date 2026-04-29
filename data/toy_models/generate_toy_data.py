"""
Utility to generate toy datasets for robustness probes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_moons, make_circles

def generate_xor(n_samples=2000, noise=0.1):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    return X, y

def generate_moons(n_samples=2000, noise=0.1):
    return make_moons(n_samples=n_samples, noise=noise)

def generate_circles(n_samples=2000, noise=0.1, factor=0.5):
    return make_circles(n_samples=n_samples, noise=noise, factor=factor)

def generate_spiral(n_samples=2000, noise=0.1):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    X = np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y))))
    y = np.hstack((np.zeros(n_samples),np.ones(n_samples)))
    return X, y

def main():
    data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    generators = {
        "xor": generate_xor,
        "moons": generate_moons,
        "circles": generate_circles,
        "spiral": generate_spiral
    }
    
    for name, gen in generators.items():
        X, y = gen()
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["y"] = y
        df.to_csv(data_dir / f"{name}.csv", index=False)
        print(f"Generated {name}.csv")

if __name__ == "__main__":
    main()
