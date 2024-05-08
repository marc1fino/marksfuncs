# importar llibreríes
import numpy as np
import matplotlib.pyplot as plt
import scipy
import io
from PIL import Image

class Linear:
# funcions
    def __init__(self, m=None, b=None, x_values_list=None):
        self.m = m
        self.b = b
        self.x_values_list= x_values_list
        if any(param is None for param in [m, b, x_values_list]):
            raise ValueError("All parameters must be provided.")
    def graph(self):
        # crear rang de valors per x
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1] , self.x_values_list[2])

        # calcular els valors de y usant la funció lineal
        # f(x) = mx + b
        y_values =  y_values = self.m * x_values + self.b

        # graph la funció lineal
        plt.figure("Linear Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.m}x + {self.b}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Linear Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        # crear rang de valors per x
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1] , self.x_values_list[2])

        # calcular els valors de y usant la funció lineal
        # f(x) = mx + b
        y_values =  y_values = self.m * x_values + self.b

        # graph la funció lineal
        plt.figure("Linear Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.m}x + {self.b}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Linear Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf


class Quadratic:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = ax ** 2 + bx + c
        y_values = self.a * x_values ** 2 + self.b * x_values + self.c

        plt.figure("Quadratic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}x^2 + {self.b}x + {self.c}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Quadratic Function')
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = ax ** 2 + bx + c
        y_values = self.a * x_values ** 2 + self.b * x_values + self.c

        plt.figure("Quadratic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}x^2 + {self.b}x + {self.c}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Quadratic Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Exponential:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(bx)
        y_values = self.a * np.exp(self.b * x_values)

        plt.figure("Exponential Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Exponential Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(bx)
        y_values = self.a * np.exp(self.b * x_values)

        plt.figure("Exponential Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Exponential Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Logarithmic:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * log(bx)
        y_values = np.empty_like(x_values)
        for i, x in enumerate(x_values):
            if x <= 0:
                y_values[i] = np.nan
            else:
                y_values[i] = self.a * np.log(self.b * x)

        plt.figure("Logarithmic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * log({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Logarithmic Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * log(bx)
        y_values = np.empty_like(x_values)
        for i, x in enumerate(x_values):
            if x <= 0:
                y_values[i] = np.nan
            else:
                y_values[i] = self.a * np.log(self.b * x)

        plt.figure("Logarithmic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * log({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Logarithmic Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Trigonometric:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * sin(bx)
        y_values = self.a * np.sin(self.b * x_values)

        plt.figure("Trigonometric Function (Senosoidal)")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sin({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Trigonometric Function (Senosoidal)')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * sin(bx)
        y_values = self.a * np.sin(self.b * x_values)

        plt.figure("Trigonometric Function (Senosoidal)")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sin({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Trigonometric Function (Senosoidal)')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Polynomial:
    def __init__(self, coeficients=None, x_values_list=None):
        self.coeficients = coeficients
        self.x_values_list = x_values_list
        if any(param is None for param in [coeficients, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.polyval(self.coeficients, x_values)

        plt.figure("Polynomial Function")
        plt.plot(x_values, y_values, label=f'f(x) = {np.poly1d(self.coeficients)}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Polynomial Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.polyval(self.coeficients, x_values)

        plt.figure("Polynomial Function")
        plt.plot(x_values, y_values, label=f'f(x) = {np.poly1d(self.coeficients)}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Polynomial Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Hyperbolic:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * sinh(bx)
        y_values = self.a * np.sinh(self.b * x_values)

        plt.figure("Hyperbolic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sinh({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Hyperbolic Function')
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * sinh(bx)
        y_values = self.a * np.sinh(self.b * x_values)

        plt.figure("Hyperbolic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sinh({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Hyperbolic Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class InverseTrigonometric:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * arcsin(bx)
        y_values = self.a * np.arcsin(self.b * x_values)

        plt.figure("Inverse Trigonometric Function (Arcosine)")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * arcsin({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Inverse Trigonometric Function (Arcosine)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * arcsin(bx)
        y_values = self.a * np.arcsin(self.b * x_values)

        plt.figure("Inverse Trigonometric Function (Arcosine)")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * arcsin({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Inverse Trigonometric Function (Arcosine)')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Rational:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a / (bx)
        y_values = self.a / (self.b * x_values)

        plt.figure("Rational Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} / ({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Rational Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a / (bx)
        y_values = self.a / (self.b * x_values)

        plt.figure("Rational Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} / ({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Rational Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Radical:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * √bx
        y_values = self.a * np.sqrt(np.maximum(self.b * x_values, 0))
        
        plt.figure("Radical Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sqrt({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Radical Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * √bx
        y_values = self.a * np.sqrt(np.maximum(self.b * x_values, 0))
        
        plt.figure("Radical Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * sqrt({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Radical Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Power:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * x**b
        y_values = self.a * (x_values ** self.b)

        plt.figure("Power Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * x^{self.b}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Power Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * x**b
        y_values = self.a * (x_values ** self.b)

        plt.figure("Power Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * x^{self.b}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Power Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Absolute:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * |bx|
        y_values = self.a * np.abs(self.b * x_values)

        plt.figure("Absolute Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * |{self.b}x|')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Absolute Function')
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * |bx|
        y_values = self.a * np.abs(self.b * x_values)

        plt.figure("Absolute Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * |{self.b}x|')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Absolute Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class UnitStep:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(X) = a * u(bx)
        y_values = self.a * np.heaviside(self.b * x_values, 0.5)

        plt.figure("Unit Step Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * u({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Unit Step Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(X) = a * u(bx)
        y_values = self.a * np.heaviside(self.b * x_values, 0.5)

        plt.figure("Unit Step Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * u({self.b}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Unit Step Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Sigmoid:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a / (1 + exp(-b * (x - c)))
        y_values = self.a / (1 + np.exp(-self.b * (x_values - self.c)))

        plt.figure("Sigmoid Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} / (1 + exp(-{self.b}(x - {self.c})))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sigmoid Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a / (1 + exp(-b * (x - c)))
        y_values = self.a / (1 + np.exp(-self.b * (x_values - self.c)))

        plt.figure("Sigmoid Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} / (1 + exp(-{self.b}(x - {self.c})))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sigmoid Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Heaviside:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = u(x)
        y_values = np.heaviside(x_values, 0.5)

        plt.figure("Heaviside Function")
        plt.plot(x_values, y_values, label='f(x) = u(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Heaviside Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = u(x)
        y_values = np.heaviside(x_values, 0.5)

        plt.figure("Heaviside Function")
        plt.plot(x_values, y_values, label='f(x) = u(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Heaviside Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Dirichlet:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sinc(ax - b)
        y_values = np.sinc(self.a * x_values - self.b)

        plt.figure("Dirichlet Function")
        plt.plot(x_values, y_values, label=f'f(x) = sinc({self.a}x - {self.b})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Dirichlet Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sinc(ax - b)
        y_values = np.sinc(self.a * x_values - self.b)

        plt.figure("Dirichlet Function")
        plt.plot(x_values, y_values, label=f'f(x) = sinc({self.a}x - {self.b})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Dirichlet Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Gauss:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(-((x - b ** 2) / (2 * c ** 2)))
        y_values = self.a * np.exp(-((x_values - self.b) ** 2) / (2 * self.c ** 2))

        plt.figure("Gauss Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp(-((x - {self.b})^2) / (2 * {self.c}^2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gauss Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(-((x - b ** 2) / (2 * c ** 2)))
        y_values = self.a * np.exp(-((x_values - self.b) ** 2) / (2 * self.c ** 2))

        plt.figure("Gauss Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp(-((x - {self.b})^2) / (2 * {self.c}^2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gauss Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Planck:
    def __init__(self, h=None, k=None, c=None, T=None, x_values_list=None):
        self.h = h
        self.k = k
        self.c = c
        self.T = T
        self.x_values_list = x_values_list
        if any(param is None for param in [h, k, c, T, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = 1 / (exp((hc) / (ktx) - 1)
        expo = self.h * self.c / (self.k * self.T * x_values)
        y_values = 1 / (np.exp(expo) - 1)

        plt.figure("Planck Function")
        plt.plot(x_values, y_values, label=f'f(x) = 1 / (exp(({self.h} * {self.c}) / ({self.k} * {self.T}x) - 1)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Planck Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = 1 / (exp((hc) / (ktx) - 1)
        expo = self.h * self.c / (self.k * self.T * x_values)
        y_values = 1 / (np.exp(expo) - 1)

        plt.figure("Planck Function")
        plt.plot(x_values, y_values, label=f'f(x) = 1 / (exp(({self.h} * {self.c}) / ({self.k} * {self.T}x) - 1)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Planck Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Triangular:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = max(0, a - |x - b| / c)
        y_values = np.maximum(0, self.a - np.abs(x_values - self.b) / self.c)

        plt.figure("Triangular Function")
        plt.plot(x_values, y_values, label=f'f(x) = max(0, {self.a} - |x - {self.b}| / {self.c})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Triangular Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = max(0, a - |x - b| / c)
        y_values = np.maximum(0, self.a - np.abs(x_values - self.b) / self.c)

        plt.figure("Triangular Function")
        plt.plot(x_values, y_values, label=f'f(x) = max(0, {self.a} - |x - {self.b}| / {self.c})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Triangular Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Sign:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sign(x)
        y_values = np.sign(x_values)

        plt.figure("Sign Function")
        plt.plot(x_values, y_values, label='f(x) = sign(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sign Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sign(x)
        y_values = np.sign(x_values)

        plt.figure("Sign Function")
        plt.plot(x_values, y_values, label='f(x) = sign(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sign Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class SawTooth:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a(x - 2π*floor(x/2π))
        period = 2 * np.pi / self.b
        y_values = self.a * (x_values - period * np.floor(x_values / period))

        plt.figure("Sawtooth Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}(x - 2π*floor(x/2π))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sawtooth Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a(x - 2π*floor(x/2π))
        period = 2 * np.pi / self.b
        y_values = self.a * (x_values - period * np.floor(x_values / period))

        plt.figure("Sawtooth Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}(x - 2π*floor(x/2π))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sawtooth Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class SquareWave:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a(1 -2(floor(x/2π) % 2))
        period = 2 * np.pi / self.b
        y_values = self.a * (1 - 2 * (np.floor(x_values / period) % 2))

        plt.figure("Square Wave Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}(1 - 2(floor(x/2pi) % 2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Square Wave Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a(1 -2(floor(x/2π) % 2))
        period = 2 * np.pi / self.b
        y_values = self.a * (1 - 2 * (np.floor(x_values / period) % 2))

        plt.figure("Square Wave Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a}(1 - 2(floor(x/2pi) % 2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Square Wave Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class SincDirichlet:
    def __init__(self, a=None, x_values_list=None):
        self.a = a
        self.x_values_list = x_values_list
        if any(param is None for param in [a, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sinc(ax)
        y_values = np.sinc(self.a * x_values)

        plt.figure("Sinc Dirichlet Function")
        plt.plot(x_values, y_values, label=f'f(x) = sinc({self.a}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sinc Dirichlet Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = sinc(ax)
        y_values = np.sinc(self.a * x_values)

        plt.figure("Sinc Dirichlet Function")
        plt.plot(x_values, y_values, label=f'f(x) = sinc({self.a}x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sinc Dirichlet Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class HyperbolicTangent:
    def __init__(self, a=None, x_values_list=None):
        self.a = a
        self.x_values_list = x_values_list
        if any(param is None for param in [a, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * tanh(x)
        y_values = self.a * np.tanh(x_values)

        plt.figure("Hyperbolic Tangent Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * tanh(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Hyperbolic Tangent Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * tanh(x)
        y_values = self.a * np.tanh(x_values)

        plt.figure("Hyperbolic Tangent Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * tanh(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Hyperbolic Tangent Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class BesselOrder0:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = J0(x)
        y_values = scipy.special.jn(0, x_values)

        plt.figure("Bessel Function of Order 0")
        plt.plot(x_values, y_values, label='f(x) = J0(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Bessel Function of Order 0")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = J0(x)
        y_values = scipy.special.jn(0, x_values)

        plt.figure("Bessel Function of Order 0")
        plt.plot(x_values, y_values, label='f(x) = J0(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Bessel Function of Order 0")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Error:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = erf(x)
        y_values = scipy.special.erf(x_values)

        plt.figure("Error Function")
        plt.plot(x_values, y_values, label='f(x) = erf(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Error Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = erf(x)
        y_values = scipy.special.erf(x_values)

        plt.figure("Error Function")
        plt.plot(x_values, y_values, label='f(x) = erf(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Error Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class SineFresnel:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = S(x)
        y_values = scipy.special.fresnel(x_values)[0]

        plt.figure("Fresnel Sine Function")
        plt.plot(x_values, y_values, label='f(x) = S(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Fresnel Sine Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = S(x)
        y_values = scipy.special.fresnel(x_values)[0]

        plt.figure("Fresnel Sine Function")
        plt.plot(x_values, y_values, label='f(x) = S(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Fresnel Sine Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class CosineFresnel:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = C(x)
        y_values = scipy.special.fresnel(x_values)[1]

        plt.figure("Fresnel Cosine Function")
        plt.plot(x_values, y_values, label='f(x) = C(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Fresnel Cosine Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = C(x)
        y_values = scipy.special.fresnel(x_values)[1]

        plt.figure("Fresnel Cosine Function")
        plt.plot(x_values, y_values, label='f(x) = C(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Fresnel Cosine Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Airy:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = Ai(x)
        y_values = scipy.special.airy(x_values)[0]

        plt.figure("Airy Function")
        plt.plot(x_values, y_values, label='f(x) = Ai(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Airy Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = Ai(x)
        y_values = scipy.special.airy(x_values)[0]

        plt.figure("Airy Function")
        plt.plot(x_values, y_values, label='f(x) = Ai(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Airy Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Logistic:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = c / (1 + exp(-b(x - a)))
        expo = -self.b * (x_values - self.a)
        y_values = self.c / (1 + np.exp(expo))

        plt.figure("Logistic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.c} / (1 + exp(-{self.b}(x - {self.a})))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Logistic Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = c / (1 + exp(-b(x - a)))
        expo = -self.b * (x_values - self.a)
        y_values = self.c / (1 + np.exp(expo))

        plt.figure("Logistic Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.c} / (1 + exp(-{self.b}(x - {self.a})))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Logistic Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Ackermann:
    def __init__(self, m=None, n=None, x_values_list=None):
        self.m = m
        self.n = n
        self.x_values_list = x_values_list
        if any(param is None for param in [m, n, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        def ack(m, n):
            if m == 0:
                return n + 1
            elif n == 0:
                return ack(m - 1, 1)
            else:
                return ack(m - 1, ack(m, n - 1))

        # Calcular algunos puntos de la función de Ackermann para valores pequeños de m y n
        m_values = np.arange(0, self.m + 1)
        n_values = np.arange(0, self.n + 1)
        ackermann_values = np.array([[ack(m, n) for n in n_values] for m in m_values])

        # Crear una malla para el gráfico
        M, N = np.meshgrid(m_values, n_values)

        # Mostrar los puntos en un gráfico de dispersión
        plt.figure("Ackermann Function")
        plt.scatter(M, N, c=ackermann_values, cmap='viridis', marker='s', s=200)
        plt.colorbar(label="Ackermann Value")
        plt.xlabel('m')
        plt.ylabel('n')
        plt.title("Ackermann Function")
        plt.grid(True)
        plt.show()
    def image(self):
        def ack(m, n):
            if m == 0:
                return n + 1
            elif n == 0:
                return ack(m - 1, 1)
            else:
                return ack(m - 1, ack(m, n - 1))

        # Calcular algunos puntos de la función de Ackermann para valores pequeños de m y n
        m_values = np.arange(0, self.m + 1)
        n_values = np.arange(0, self.n + 1)
        ackermann_values = np.array([[ack(m, n) for n in n_values] for m in m_values])

        # Crear una malla para el gráfico
        M, N = np.meshgrid(m_values, n_values)

        # Mostrar los puntos en un gráfico de dispersión
        plt.figure("Ackermann Function")
        plt.scatter(M, N, c=ackermann_values, cmap='viridis', marker='s', s=200)
        plt.colorbar(label="Ackermann Value")
        plt.xlabel('m')
        plt.ylabel('n')
        plt.title("Ackermann Function")
        plt.grid(True)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Fibonacci:
    def __init__(self, n=None, x_values_list=None):
        self.n = n
        self.x_values_list = x_values_list
        if any(param is None for param in [n, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        def fib(n):
            if n <= 1:
                return n
            else:
                return fib(n - 1) + fib(n - 2)

        # Calcular algunos puntos de la función de Fibonacci para valores pequeños de n
        n_values = np.arange(0, self.n + 1)
        fibonacci_values = np.array([fib(n) for n in n_values])

        # Mostrar los puntos en un gráfico de dispersión
        plt.figure("Fibonacci Function")
        plt.plot(n_values, fibonacci_values, marker='o', linestyle='-')
        plt.xlabel('n')
        plt.ylabel('Fib(n)')
        plt.title('Fibonacci Function')
        plt.grid(True)
        plt.show()
    def image(self):
        def fib(n):
            if n <= 1:
                return n
            else:
                return fib(n - 1) + fib(n - 2)

        # Calcular algunos puntos de la función de Fibonacci para valores pequeños de n
        n_values = np.arange(0, self.n + 1)
        fibonacci_values = np.array([fib(n) for n in n_values])

        # Mostrar los puntos en un gráfico de dispersión
        plt.figure("Fibonacci Function")
        plt.plot(n_values, fibonacci_values, marker='o', linestyle='-')
        plt.xlabel('n')
        plt.ylabel('Fib(n)')
        plt.title('Fibonacci Function')
        plt.grid(True)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Legendre:
    def __init__(self, grade=None, x_values_list=None):
        self.grade = grade
        self.x_values_list = x_values_list
        if any(param is None for param in [grade, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = P_n(x)
        y_values = scipy.special.eval_legendre(int(self.grade), x_values)

        plt.figure(f"Legendre Function of Grade {self.grade}")
        plt.plot(x_values, y_values, label=f'P_{self.grade}(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f"Legendre Function of Grade {self.grade}")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = P_n(x)
        y_values = scipy.special.eval_legendre(int(self.grade), x_values)

        plt.figure(f"Legendre Function of Grade {self.grade}")
        plt.plot(x_values, y_values, label=f'P_{self.grade}(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f"Legendre Function of Grade {self.grade}")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class Gaussian:
    def __init__(self, mu=None, sigma=None, x_values_list=None):
        self.mu = mu
        self.sigma = sigma
        self.x_values_list = x_values_list
        if any(param is None for param in [mu, sigma, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = e ** (-((x - mu ** 2) / (2 * sigma ** 2)) / (sigma * √2π))
        y_values = np.exp(-((x_values - self.mu) ** 2) / (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))

        plt.figure("Gaussian Function")
        plt.plot(x_values, y_values, label=f'f(x) = e^(-((x - {self.mu})^2) / (2 * {self.sigma}^2)) / ({self.sigma} * sqrt(2 * π))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gaussian Function')
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = e ** (-((x - mu ** 2) / (2 * sigma ** 2)) / (sigma * √2π))
        y_values = np.exp(-((x_values - self.mu) ** 2) / (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))

        plt.figure("Gaussian Function")
        plt.plot(x_values, y_values, label=f'f(x) = e^(-((x - {self.mu})^2) / (2 * {self.sigma}^2)) / ({self.sigma} * sqrt(2 * π))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gaussian Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class GeneralizedLogistics:
    def __init__(self, a=None, b=None, c=None, d=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, d, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = d / (1 + exp(-b * (x - a)) + c)
        expo = -self.b * (x_values - self.a)
        y_values = self.d / (1 + np.exp(expo)) + self.c

        plt.figure("Generalized Logistics Functions")
        plt.plot(x_values, y_values, label=f'f(x) = {self.d} / (1 + exp(-{self.b}(x - {self.a}))) + {self.c}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Generalized Logistics Functions')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = d / (1 + exp(-b * (x - a)) + c)
        expo = -self.b * (x_values - self.a)
        y_values = self.d / (1 + np.exp(expo)) + self.c

        plt.figure("Generalized Logistics Functions")
        plt.plot(x_values, y_values, label=f'f(x) = {self.d} / (1 + exp(-{self.b}(x - {self.a}))) + {self.c}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Generalized Logistics Functions')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class LogSineProduct:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = log|sin(x)|
        y_values = np.log(np.abs(np.sin(x_values)))

        plt.figure("Sine Product Logarithm Function")
        plt.plot(x_values, y_values, label='f(x) = log|sin(x)|')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sine Product Logarithm Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = log|sin(x)|
        y_values = np.log(np.abs(np.sin(x_values)))

        plt.figure("Sine Product Logarithm Function")
        plt.plot(x_values, y_values, label='f(x) = log|sin(x)|')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Sine Product Logarithm Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class LogGamma:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = ln(Γ(x))
        y_values = scipy.special.gammaln(x_values)

        plt.figure("Gamma Logarithm Function")
        plt.plot(x_values, y_values, label='f(x) = ln(Γ(x))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gamma Logarithm Function')
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = ln(Γ(x))
        y_values = scipy.special.gammaln(x_values)

        plt.figure("Gamma Logarithm Function")
        plt.plot(x_values, y_values, label='f(x) = ln(Γ(x))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gamma Logarithm Function')
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class GeneralizedExponential:
    def __init__(self, a=None, b=None, c=None, x_values_list=None):
        self.a = a
        self.b = b
        self.c = c
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, c, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(-(x - b) ** 2 / (2c ** 2))
        y_values = self.a * np.exp(-(x_values - self.b)**2 / (2 * self.c**2))

        plt.figure("Generalized Exponential Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp(-((x - {self.b})^2) / (2 * {self.c}^2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Generalized Exponential Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(-(x - b) ** 2 / (2c ** 2))
        y_values = self.a * np.exp(-(x_values - self.b)**2 / (2 * self.c**2))

        plt.figure("Generalized Exponential Function")
        plt.plot(x_values, y_values, label=f'f(x) = {self.a} * exp(-((x - {self.b})^2) / (2 * {self.c}^2))')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Generalized Exponential Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class ModifiedAckermann:
    def __init__(self, m=None, n=None, x_values_list=None):
        self.m = m
        self.n = n
        self.x_values_list = x_values_list
        if any(param is None for param in [m, n, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        def ack_modificada(m, n):
            if m == 0:
                return n + 1
            elif n == 0:
                return ack_modificada(m - 1, 1)
            elif m == 1:
                return n + 2
            elif m == 2:
                return 2 * n + 3
            elif m == 3:
                return 2 ** (n + 3) - 3
            else:
                raise ValueError("La funció modificada d'Ackermann només es defineix per m = 0, 1, 2, 3.")

        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = Ack_mod(m, x)
        y_values = np.array([ack_modificada(int(self.m), int(n)) for n in x_values])

        plt.figure("Modified Akermann Function")
        plt.plot(x_values, y_values, label=f'Ack_mod({self.m}, x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Modified Akermann Function")
        plt.grid(True)
        plt.legend()
        plt.show()
def image(self):
        def ack_modificada(m, n):
            if m == 0:
                return n + 1
            elif n == 0:
                return ack_modificada(m - 1, 1)
            elif m == 1:
                return n + 2
            elif m == 2:
                return 2 * n + 3
            elif m == 3:
                return 2 ** (n + 3) - 3
            else:
                raise ValueError("La funció modificada d'Ackermann només es defineix per m = 0, 1, 2, 3.")

        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = Ack_mod(m, x)
        y_values = np.array([ack_modificada(int(self.m), int(n)) for n in x_values])

        plt.figure("Modified Akermann Function")
        plt.plot(x_values, y_values, label=f'Ack_mod({self.m}, x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Modified Akermann Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Floor:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) =  floor(x)
        y_values = np.floor(x_values)
        
        plt.figure("Floor Function")
        plt.plot(x_values, y_values, label='floor(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Floor Function")
        plt.grid(True)
        plt.legend()
        plt.show()
def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) =  floor(x)
        y_values = np.floor(x_values)
        
        plt.figure("Floor Function")
        plt.plot(x_values, y_values, label='floor(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Floor Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Ceil:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.ceil(x_values)
        
        plt.figure("Ceil Function")
        plt.plot(x_values, y_values, label='ceil(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Ceil Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.ceil(x_values)
        
        plt.figure("Ceil Function")
        plt.plot(x_values, y_values, label='ceil(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Ceil Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class Identity:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = x
        y_values = x_values
        
        plt.figure("Identity Function")
        plt.plot(x_values, y_values, label='f(x) = x')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Identity Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = x
        y_values = x_values
        
        plt.figure("Identity Function")
        plt.plot(x_values, y_values, label='f(x) = x')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Identity Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class BesselOrder1:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = J1(x)
        y_values = scipy.special.jv(1, x_values)
        
        plt.figure("Bessel Function of Order 1")
        plt.plot(x_values, y_values, label='J1(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Bessel Function of Order 1")
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = J1(x)
        y_values = scipy.special.jv(1, x_values)
        
        plt.figure("Bessel Function of Order 1")
        plt.plot(x_values, y_values, label='J1(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Bessel Function of Order 1")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf

class DiracDelta:
    def __init__(self, x_values_list=None):
        self.x_values_list = x_values_list
        if x_values_list is None:
            raise ValueError("The range of values ​​of x must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = δ(x)
        y_values = np.zeros_like(x_values)
        y_values[len(x_values) // 2] = 1

        plt.figure("Dirac Delta Function")
        plt.stem(x_values, y_values, label='δ(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Dirac Delta Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = δ(x)
        y_values = np.zeros_like(x_values)
        y_values[len(x_values) // 2] = 1

        plt.figure("Dirac Delta Function")
        plt.stem(x_values, y_values, label='δ(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Dirac Delta Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class ComplexExponential:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(j * bx)
        y_values = self.a * np.exp(1j * self.b * x_values)

        plt.figure("Complex Exponential Function")
        plt.plot(x_values, np.real(y_values), label=f'Real: {self.a} * exp(j * {self.b} * x)')
        plt.plot(x_values, np.imag(y_values), label=f'Imaginary: {self.a} * exp(j * {self.b} * x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Complex Exponential Function")
        plt.grid(True)
        plt.legend()
        plt.show()

    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        # f(x) = a * exp(j * bx)
        y_values = self.a * np.exp(1j * self.b * x_values)

        plt.figure("Complex Exponential Function")
        plt.plot(x_values, np.real(y_values), label=f'Real: {self.a} * exp(j * {self.b} * x)')
        plt.plot(x_values, np.imag(y_values), label=f'Imaginary: {self.a} * exp(j * {self.b} * x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Complex Exponential Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
class TopHat:
    def __init__(self, a=None, b=None, x_values_list=None):
        self.a = a
        self.b = b
        self.x_values_list = x_values_list
        if any(param is None for param in [a, b, x_values_list]):
            raise ValueError("All parameters must be provided.")

    def graph(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.where((x_values >= self.a) & (x_values <= self.b), 1, 0)

        plt.figure("Top Hat Function")
        plt.plot(x_values, y_values, label=f'Caja({self.a}, {self.b})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Top Hat Function")
        plt.grid(True)
        plt.legend()
        plt.show()
    def image(self):
        x_values = np.linspace(self.x_values_list[0], self.x_values_list[1], self.x_values_list[2])
        y_values = np.where((x_values >= self.a) & (x_values <= self.b), 1, 0)

        plt.figure("Top Hat Function")
        plt.plot(x_values, y_values, label=f'Caja({self.a}, {self.b})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Top Hat Function")
        plt.grid(True)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        
        return img_buf
