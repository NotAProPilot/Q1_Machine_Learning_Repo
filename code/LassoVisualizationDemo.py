import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate a small 2D regression dataset
X, y, coef = make_regression(
    n_samples=100,
    n_features=2,
    noise=10,
    coef=True,
    random_state=42
)

# Fit plain linear regression for baseline
lr = LinearRegression().fit(X, y)
coef_est = lr.coef_

# Create grid for coefficients
beta1 = np.linspace(-50, 50, 200)
beta2 = np.linspace(-50, 50, 200)
B1, B2 = np.meshgrid(beta1, beta2)

# Compute a simple squared error loss surface (ignoring intercept)
loss_surface = np.mean((y.reshape(-1,1,1) - (X[:,0].reshape(-1,1,1)*B1 + X[:,1].reshape(-1,1,1)*B2))**2, axis=0)

# L1 and L2 norms for constraint boundaries
L1_norm = np.abs(B1) + np.abs(B2)
L2_norm = np.sqrt(B1**2 + B2**2)

# Create interactive figure
fig = go.Figure()

# Loss contours
fig.add_trace(go.Contour(
    x=beta1, y=beta2, z=loss_surface,
    contours=dict(start=500, end=2000, size=200, coloring='lines'),
    showscale=False,
    line=dict(color="gray"),
    name="Loss contours"
))

# Add L1 norm diamond (constraint boundary at some c)
c1 = 40
fig.add_trace(go.Contour(
    x=beta1, y=beta2, z=L1_norm,
    contours=dict(start=c1, end=c1, size=1, coloring='lines'),
    line=dict(color="blue", width=3),
    showscale=False,
    name="L1 constraint"
))

# Add L2 norm circle (constraint boundary at some c)
c2 = 40
fig.add_trace(go.Contour(
    x=beta1, y=beta2, z=L2_norm,
    contours=dict(start=c2, end=c2, size=1, coloring='lines'),
    line=dict(color="red", width=3),
    showscale=False,
    name="L2 constraint"
))

# Add estimated coefficients point
fig.add_trace(go.Scatter(
    x=[coef_est[0]], y=[coef_est[1]],
    mode="markers+text",
    text=["OLS Solution"],
    textposition="top center",
    marker=dict(size=10, color="black"),
    name="OLS Coeffs"
))

fig.update_layout(
    title="Loss Surface with L1 (diamond) and L2 (circle) Norm Constraints",
    xaxis_title="β1",
    yaxis_title="β2",
    width=800, height=600
)

fig.show()
