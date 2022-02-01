if __name__ == '__main__':
    print("pro2")


# 8.1
# [E], [S], [ES], [P]  are the concentration of E, S, ES, P respectively.
# As for the law of mass action, we can get 4 equations:
# d[E]/dt = (k2+k3)[ES] - k1[E][S]
# d[S]/dt = k2[ES] - k1[E][S]
# d[ES]/dt = k1[E][S] - (k2+k3)[ES]
# d[P]/dt = k3[ES]

# 8.2
# As for the queastion, we can also get these 2 equations:
# [E] = 1-[ES]
# [P] = 10-[S]-[ES]
# Unfortunately, I did not solve this problem in the end.
# Mainly because I could not determine the relationship between the concentration of E and the concentration of S.
# Therefore, I was unable to convert the four equations listed earlier into the form d[...]/dt and finally unravel it.
# If I can get their relationship, I can solve this using the Runge Kutta method by the code below.
# I will continue to think about this issue.
# I hope that I can learn relevant knowledge in the study of this project in the future.

# Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"
def dydx(x, y):
    return (x - y) / 2


# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):
    # Count number of iterations using step size or
    # step height h
    n = (int)((x - x0) / h)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y


# Driver method
x0 = 0
y0 = 1
x = 2
h = 0.2
print('The value of y at x is:', rungeKutta(x0, y0, x, h))

# 8.3

# d[P]/dt = k3[ES] = k3(1-[E])
# Differentiate both sides of the equation for the concentration of S at the same time.
# But since I can't determine the relationship between E concentration and S concentration, I also can't solve 8.3.
