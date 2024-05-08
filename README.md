# marksfuncs

A mathematical library to graph many mathematical functions in a plot.

Developed by Marc Pérez (c) 2024

## Examples of How To Use

Watch a function's parameters

```python
from marksfuncs import Linear

# For Printing Parameters In Console
print(Linear())

# You Can Add It In Anything That Supports An String
message = "I like bananas "
print(message+Linear())
```

Make The Grafic Of Your Function

```python
from marksfuncs import Exponential

# Printing A Message When The Grafic Finished Showing
print(Exponential(a=6, b=5, x_values_list=[-5, 5, 100]))

# Only Show The Grafic
Exponential(a=4, b=2, x_values_list=[-5, 5, 100])
```

Graphics Explanation

![Graphic Example](https://i.ibb.co/fq240nZ/imagen-2024-05-07-205144279.png)

```python
from marksfuncs import Exponential
"""

x_values_list = A list to indicate de values of x:
    x_values_list[-5, 5, 100] = x will start at number -5 and end in 5 leaving 100 spaces.

a, b (This values depend on the type of function we would like to graph) = In this case,
Exponential Function Formula = f(x) = a exp(b x)

<------------------------------------------------------------------------------------->

You can navigate through the graphic to know the exact values of y and x in the position you click,
or just can watch the numbers of the graphic. In the bottom you can save the graphic or move it,
in every graphic there's a label that tells you the current formula of the function.

"""
# Create The Graphic
Exponential(a=3, b=4, x_values_list=[-5, 5, 100]).graph()
```
