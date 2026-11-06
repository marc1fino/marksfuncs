# marksfuncs

A comprehensive mathematical function plotting library for Python with **50+ built-in function classes**, **free-form expression evaluation**, and **publication-quality graphs**.

The project provides ready-to-use classes for elementary functions (linear, quadratic, exponential), transcendental functions (trigonometric, hyperbolic, logarithmic), special functions (Bessel, Gamma, Error, Airy), probability distributions (Gaussian, Weibull, Beta, Gamma, Poisson), and a powerful free-form expression evaluator that works like GeoGebra — all with clean APIs for both interactive plotting and programmatic image generation.

**Python** | **Mathematics** | **Data Visualization** | **Scientific Computing**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%2C%20SciPy%2C%20Matplotlib-blue?style=flat-square)

## UI

This library also features a companion graphical interface designed for seamless interaction, real-time calculations, and effortless mathematical exploration.

- **Repository:** [marksfuncs-ui](https://github.com/marc1fino/marksfuncs-ui)
- **Highlights:**
  * **Interactive Plotting:** Visualize and analyze mathematical functions and expressions easily.
  * **Real-time Results:** Instant calculations and evaluations as you type.
  * **Lightweight & Cross-Platform:** Built to complement the core Python library smoothly.

## Features

- **50+ function classes** — from elementary to special functions and probability distributions
- **Free-form expressions** — `ExpressionFunction` evaluates any mathematical string (like GeoGebra)
- **Polynomial by coefficients** — `PolynomialCoefficients` defines polynomials via degree→coefficient mapping
- **Dual output modes** — `graph()` for interactive display, `image()` for programmatic PNG buffers
- **Customizable ranges** — configurable x-range (start, end, points) and optional y-axis limits
- **Rich label support** — automatic formula labels on every plot
- **Publication ready** — high-quality Matplotlib output with grid, legend, and axis labels
- **Zero-config** — works out of the box with sensible defaults

## Function Categories

### Elementary Functions

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Identity` | f(x) = x | `x_values_list` |
| `Linear` | f(x) = mx + b | `m`, `b`, `x_values_list` |
| `Quadratic` | f(x) = ax² + bx + c | `a`, `b`, `c`, `x_values_list`, `y_values?` |
| `Polynomial` | f(x) = Σ aᵢxⁱ (from string) | `polynomial_str`, `x_values_list`, `y_values?` |
| `PolynomialCoefficients` | f(x) = Σ aᵢxⁱ (from dict/list) | `coefficients`, `x_values_list`, `y_values?` |
| `Power` | f(x) = axᵇ | `a`, `b`, `x_values_list`, `y_values?` |
| `Root` / `Radical` | f(x) = a√(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `Absolute` | f(x) = a\|bx\| | `a`, `b`, `x_values_list`, `y_values?` |

### Exponential & Logarithmic

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Exponential` | f(x) = a·eᵇˣ | `a`, `b`, `x_values_list`, `y_values?` |
| `Logarithmic` | f(x) = a·log(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `GeneralizedExponential` | f(x) = a·e^(-(x-b)²/2c²) | `a`, `b`, `c`, `x_values_list` |
| `ComplexExponential` | f(x) = a·e^(jbx) | `a`, `b`, `x_values_list` |

### Trigonometric & Hyperbolic

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Trigonometric` | f(x) = a·sin(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `Hyperbolic` | f(x) = a·sinh(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `HyperbolicTangent` | f(x) = a·tanh(x) | `a`, `x_values_list`, `y_values?` |
| `InverseTrigonometric` | f(x) = a·arcsin(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `SincDirichlet` | f(x) = sinc(ax) | `a`, `x_values_list`, `y_values?` |
| `Dirichlet` | f(x) = sinc(ax - b) | `a`, `b`, `x_values_list`, `y_values?` |

### Special Functions

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `BesselOrder0` | f(x) = J₀(x) | `x_values_list`, `y_values?` |
| `BesselOrder1` | f(x) = J₁(x) | `x_values_list`, `y_values?` |
| `Error` | f(x) = erf(x) | `x_values_list`, `y_values?` |
| `Airy` | f(x) = Ai(x) | `x_values_list`, `y_values?` |
| `SineFresnel` | f(x) = S(x) | `x_values_list`, `y_values?` |
| `CosineFresnel` | f(x) = C(x) | `x_values_list`, `y_values?` |
| `LogGamma` | f(x) = ln(Γ(x)) | `x_values_list`, `y_values?` |
| `LogSineProduct` | f(x) = log\|sin(x)\| | `x_values_list` |

### Sigmoid & Activation Functions

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Sigmoid` | f(x) = a/(1+e^(-b(x-c))) | `a`, `b`, `c`, `x_values_list`, `y_values?` |
| `Logistic` | f(x) = c/(1+e^(-b(x-a))) | `a`, `b`, `c`, `x_values_list` |
| `GeneralizedLogistics` | f(x) = d/(1+e^(-b(x-a)))+c | `a`, `b`, `c`, `d`, `x_values_list` |
| `Gompertz` | f(x) = a·e^(-b·e^(-cx)) | `a`, `b`, `c`, `x_values_list`, `y_values?` |
| `Hill` | f(x) = Vₘₐₓ·xⁿ/(Kⁿ+xⁿ) | `vmax`, `k`, `n`, `x_values_list`, `y_values?` |

### Probability Distributions

| Class | Distribution | Parameters |
| :--- | :--- | :--- |
| `Gaussian` / `Gauss` | Normal (μ, σ) | `mu`, `sigma` / `a`, `b`, `c`, `x_values_list`, `y_values?` |
| `Weibull` | Weibull (k, λ) | `k`, `lambda_param`, `x_values_list`, `y_values?` |
| `Beta` | Beta (α, β) | `alpha`, `beta`, `x_values_list`, `y_values?` |
| `Gamma` | Gamma (k, θ) | `k`, `theta`, `x_values_list`, `y_values?` |
| `Poisson` | Poisson (λ) | `lambda_param`, `x_values_list`, `y_values?` |
| `ChiSquare` | χ² (df) | `df`, `x_values_list`, `y_values?` |

### Waveforms & Periodic Functions

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `SawTooth` | f(x) = a(x - 2π⌊x/2π⌋) | `a`, `b`, `x_values_list`, `y_values?` |
| `SquareWave` | f(x) = a(1-2⌊x/2π⌋%2) | `a`, `b`, `x_values_list`, `y_values?` |
| `Sign` | f(x) = sign(x) | `x_values_list`, `y_values?` |
| `Floor` | f(x) = ⌊x⌋ | `x_values_list` |
| `Ceil` | f(x) = ⌈x⌉ | `x_values_list` |
| `Triangular` | f(x) = max(0, a - \|x-b\|/c) | `a`, `b`, `c`, `x_values_list`, `y_values?` |

### Step & Impulse Functions

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `UnitStep` | f(x) = a·u(bx) | `a`, `b`, `x_values_list`, `y_values?` |
| `Heaviside` | f(x) = u(x) | `x_values_list`, `y_values?` |
| `DiracDelta` | f(x) = δ(x) | `x_values_list` |
| `TopHat` | f(x) = 1 for x∈[a,b] | `a`, `b`, `x_values_list` |

### Rational & Algebraic

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Rational` | f(x) = a/(bx) | `a`, `b`, `x_values_list`, `y_values?` |

### Physics & Engineering

| Class | Formula | Parameters |
| :--- | :--- | :--- |
| `Planck` | Blackbody radiation | `h`, `k`, `c`, `T`, `x_values_list`, `y_values?` |

### Number-Theoretic & Recursive

| Class | Description | Parameters |
| :--- | :--- | :--- |
| `Ackermann` | Ackermann function (scatter) | `m`, `n`, `x_values_list` |
| `ModifiedAckermann` | Modified Ackermann (m=0..3) | `m`, `n`, `x_values_list` |
| `Fibonacci` | Fibonacci sequence (scatter) | `n`, `x_values_list` |
| `Legendre` | Legendre polynomial Pₙ(x) | `grade`, `x_values_list` |

### Free-Form Expression Evaluator

| Class | Description | Parameters |
| :--- | :--- | :--- |
| `ExpressionFunction` | Evaluate any math expression string | `expression`, `x_values_list`, `y_values?`, `variable='x'` |

## Quick Start

```bash
pip install marksfuncs
```

```python
from marksfuncs import Exponential, Linear, PolynomialCoefficients, ExpressionFunction
from PIL import Image

# 1. Interactive plotting
Exponential(a=4, b=2, x_values_list=[-5, 5, 100]).graph()

# 2. With custom y-axis limits
Exponential(a=4, b=2, x_values_list=[-5, 5, 100], y_values=(-10, 10)).graph()

# 3. Programmatic image generation (PNG buffer)
img_buffer = Linear(m=3, b=5, x_values_list=[-5, 5, 100]).image()
img = Image.open(img_buffer)
img.save("linear.png")

# 4. Polynomial by coefficients (degree → coefficient)
# f(x) = 3x² + 2x + 1
poly = PolynomialCoefficients(
    coefficients={0: 1, 1: 2, 2: 3},  # or [1, 2, 3]
    x_values_list=[-5, 5, 100]
)
poly.graph()

# 5. Free-form expressions like GeoGebra!
expr = ExpressionFunction(
    expression="sin(x) + x**2",
    x_values_list=[-5, 5, 100]
)
expr.graph()

# Custom variable name
expr2 = ExpressionFunction(
    expression="exp(-t)*cos(t)",
    x_values_list=[0, 10, 200],
    variable='t'
)
expr2.graph()

# Advanced: special functions
expr3 = ExpressionFunction(
    expression="erf(x) + gamma(x+1)",
    x_values_list=[0.1, 5, 100]
)
expr3.graph()
```

## Installation

```bash
# From PyPI (when published)
pip install marksfuncs

# From source
git clone https://github.com/marc1fino/marksfuncs.git
cd marksfuncs
pip install -e .
```

Requires **Python 3.8+**. Dependencies: `numpy`, `scipy`, `matplotlib`, `pillow`.

## Core API

### Common Parameters

All classes share these parameters:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `x_values_list` | `[float, float, int]` | **Required.** `[x_min, x_max, num_points]` — defines the x-axis sampling range |
| `y_values` | `(float, float)` | **Optional.** `(y_min, y_max)` — sets fixed y-axis limits for consistent scaling |

### Core Methods

Every class provides:

| Method | Returns | Description |
| :--- | :--- | :--- |
| `graph()` | `None` | Displays interactive Matplotlib window with the plot |
| `image()` | `io.BytesIO` | Returns PNG image buffer for programmatic use (save, embed, serve) |

### ExpressionFunction Namespace

`ExpressionFunction` provides a safe evaluation context with:

- **Constants**: `pi`, `e`, `inf`, `nan`
- **NumPy functions**: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `log2`, `sqrt`, `cbrt`, `abs`, `floor`, `ceil`, `round`, `sign`, `heaviside`, `maximum`, `minimum`, `clip`, `power`, `mod`, `sinc`, `deg2rad`, `rad2deg`
- **SciPy special**: `erf`, `erfc`, `gamma`, `gammaln`, `beta`, `betaln`, `j0`, `j1`, `jn`, `yn`, `airy`

## Examples

### Basic Elementary Functions

```python
from marksfuncs import Linear, Quadratic, Power, Absolute

# Linear: f(x) = 2x + 3
Linear(m=2, b=3, x_values_list=[-10, 10, 200]).graph()

# Quadratic: f(x) = x² - 4x + 4
Quadratic(a=1, b=-4, c=4, x_values_list=[-2, 6, 200]).graph()

# Power: f(x) = 2x³
Power(a=2, b=3, x_values_list=[-3, 3, 200]).graph()

# Absolute value: f(x) = |x - 2|
Absolute(a=1, b=1, x_values_list=[-5, 5, 200]).graph()
```

### Example: Trigonometric & Hyperbolic

```python
from marksfuncs import Trigonometric, Hyperbolic, HyperbolicTangent

# Sine wave: f(x) = 3sin(2x)
Trigonometric(a=3, b=2, x_values_list=[-2*3.14, 2*3.14, 400]).graph()

# Hyperbolic sine: f(x) = 2sinh(x)
Hyperbolic(a=2, b=1, x_values_list=[-3, 3, 200]).graph()

# Tanh activation: f(x) = tanh(x)
HyperbolicTangent(a=1, x_values_list=[-5, 5, 200]).graph()
```

### Example: Probability Distributions

```python
from marksfuncs import Gaussian, Weibull, Beta, Gamma, Poisson, ChiSquare

# Normal distribution: μ=0, σ=1
Gaussian(mu=0, sigma=1, x_values_list=[-4, 4, 400]).graph()

# Weibull: k=2, λ=1.5
Weibull(k=2, lambda_param=1.5, x_values_list=[0, 5, 400]).graph()

# Beta: α=2, β=5
Beta(alpha=2, beta=5, x_values_list=[0, 1, 400]).graph()

# Chi-square: df=3
ChiSquare(df=3, x_values_list=[0, 10, 400]).graph()
```

### Example: Special Functions

```python
from marksfuncs import BesselOrder0, BesselOrder1, Error, Airy, LogGamma

# Bessel J₀
BesselOrder0(x_values_list=[0, 20, 400]).graph()

# Error function
Error(x_values_list=[-3, 3, 400]).graph()

# Airy Ai(x)
Airy(x_values_list=[-10, 10, 400]).graph()

# Log Gamma
LogGamma(x_values_list=[0.1, 5, 400]).graph()
```

### Free-Form Expressions (GeoGebra-like)

```python
from marksfuncs import ExpressionFunction

# Any valid mathematical expression
ExpressionFunction(
    expression="sin(x)*exp(-x/5)",
    x_values_list=[0, 20, 400]
).graph()

# Multiple terms
ExpressionFunction(
    expression="x**3 - 3*x**2 + 2*x - 1",
    x_values_list=[-2, 4, 400]
).graph()

# With special functions
ExpressionFunction(
    expression="erf(x) + sin(x**2)/2",
    x_values_list=[-3, 3, 400]
).graph()

# Custom variable
ExpressionFunction(
    expression="cos(omega*t)*exp(-t/tau)",
    x_values_list=[0, 10, 400],
    variable='t'
).graph()

# Polar-like: r = 1 + cos(theta)
ExpressionFunction(
    expression="1 + cos(x)",
    x_values_list=[0, 2*3.14159, 400]
).graph()
```

### Polynomial by Coefficients

```python
from marksfuncs import PolynomialCoefficients

# f(x) = 4x⁴ - 3x³ + 2x² - x + 5
# Using dict (degree → coefficient)
poly1 = PolynomialCoefficients(
    coefficients={0: 5, 1: -1, 2: 2, 3: -3, 4: 4},
    x_values_list=[-2, 2, 400]
)
poly1.graph()

# Using list (index = degree)
poly2 = PolynomialCoefficients(
    coefficients=[5, -1, 2, -3, 4],
    x_values_list=[-2, 2, 400]
)
poly2.graph()

# Sparse polynomial: f(x) = x¹⁰ - 1
poly3 = PolynomialCoefficients(
    coefficients={0: -1, 10: 1},
    x_values_list=[-1.5, 1.5, 400]
)
poly3.graph()
```

### Image Generation for Web/Apps

```python
from marksfuncs import Sigmoid, ExpressionFunction
from PIL import Image
import io

# Generate PNG for embedding
buf = Sigmoid(a=1, b=1, c=0, x_values_list=[-10, 10, 400]).image()

# Save to file
with open("sigmoid.png", "wb") as f:
    f.write(buf.getvalue())

# Or open directly
img = Image.open(buf)
img.show()

# Expression function to image
buf2 = ExpressionFunction(
    expression="1/(1+exp(-x))",
    x_values_list=[-10, 10, 400]
).image()
```

## Project Structure

```text
marksfuncs/
├── marksfuncs/
│   ├── __init__.py          # Exports all 50+ classes
│   └── functions.py         # All function implementations
├── setup.py                 # Packaging configuration
├── README.md                # This file
└── LICENSE                  # MIT License
```

## Requirements

- **Python 3.8+**
- `numpy` ≥ 1.21
- `scipy` ≥ 1.7
- `matplotlib` ≥ 3.5
- `pillow` ≥ 9.0

## Development

```bash
git clone https://github.com/marc1fino/marksfuncs.git
cd marksfuncs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .

# Quick smoke test
python -c "
from marksfuncs import Linear, ExpressionFunction, PolynomialCoefficients
import numpy as np

# Test Linear
assert Linear(m=2, b=3, x_values_list=[0, 1, 2])._evaluate(5) == 13

# Test ExpressionFunction
expr = ExpressionFunction('x**2 + 1', [0, 1, 2])
assert abs(expr._evaluate(3) - 10) < 1e-10

# Test PolynomialCoefficients
poly = PolynomialCoefficients([1, 2, 3], [0, 1, 2])  # 3x²+2x+1
assert poly._evaluate(2) == 17

print('All tests passed!')
"
```

## Version History

- **1.0.0** — Initial release with 40+ function classes
- **1.1.0** — Added `PolynomialCoefficients` and `ExpressionFunction` classes; enhanced special functions coverage

## Roadmap

- Parametric and polar plotting support
- 3D surface plotting (`plot_surface`)
- Animation support for time-varying functions
- LaTeX formula rendering in labels
- Export to SVG/PDF formats
- Interactive widgets (ipywidgets integration)
- Type hints and full mypy compatibility
- Comprehensive test suite with pytest

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

## Author

**Marc Pérez** — [GitHub](https://github.com/marc1fino)

---

*Mathematical function plotting for education, research, and engineering.*
