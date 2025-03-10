# Collocations.jl
Provides a collection of helpers for collocation methods in solving discrete time Bellman equations.
The math backgrounds are explained in detail in the 2nd half of this README file.


## Installation

```julia
add "https://github.com/Clpr/Collocations.jl.git#main"
```

## Usage

**Generic usage**:
```julia
import Collocations as col

?col.MarkovCollocationUniform
```


**Example**: Stochastic growth model, value function iteration

Check the notebook `example/example - vfi - growth.ipynb`







## License

MIT license.



# Math: Collocation method for the dynamic programmings with Markovian uncertainty

## Math problem

Consider the following generic value function iteration (VFI) problem:

$$
\begin{align}
& v(x,z) = \max_{c} u(x,z;c) + \beta \mathbb{E} \{ v(x',z') | z \}  &\text{Bellman equation} \\
\text{s.t. }&   x' = \mathfrak{X}(x,z,c)   & \text{State equations}  \nonumber\\
& g(x,z,c) \leq \mathbf{0}_{L}                & \text{Admissibility} \nonumber \\
& x \in \mathbb{R}^D, z \in \mathbb{R}^J, c \in \mathbb{R}^M & \text{State constraints}  \nonumber\\
& z' \sim \text{MarkovChain}_K  & \text{Exogenous states} \nonumber \\
\end{align}
$$

where

- $D$ is the dimensionality of the endogeneous states $x$
- $J$ is the dimensionality of the exogenous states $z$ which follow a $K$-state finite Markov chain $ \text{MC}_K := (\mathbf{Z},\mathbf{P}^z)$, where $\mathbf{Z} := [z^1,z^2,\dots,z^K] \in \mathbb{R}^{J,K}$ is the stacking of the states; $\mathbf{P}^z \in\mathbb{R}^{K \times K}$ is the row-to-column transition probability matrix.
- For convenience, we denote the $i$-th row of $\mathbf{P}^z$ as $\mathbf{P}^z_i \in\mathbb{R}^{1\times K}$; and the $i$-th row $j$-th column element as $\mathbf{p}^z_{i,j} \in [0,1]$.
- $g(x,z,c) \in\mathbb{R}^{L}$ is a collection of optimization constraints
- $u(x,z,c) \mapsto \mathbb{R}$ is the instantaneous utility/pay-off function
- Without special declare, we use subscripts to index a point in stackings, and use subscripts to denote states or grid node points.
- Without special declare, a vector space $\mathbb{R}^D$ is equivalent to a space of column vectors $\mathbb{R}^{D\times 1}$.

Suppose that we have found the optimal control $c^* = \mathfrak{C}(x,z)$ under the $g(x,z,c)$ constraints where $\mathfrak{C}(x,z)$ is the policy function profile mapping to an $M$-dimensional vector $c^*$.
By recursively applying the policy operator, the above dynamic programming can be written as:

$$
\begin{align}
& v(x,z) = u(x,z) + \beta \cdot \bar{v}\left(  \mathfrak{X}(x,z) ,  z \right)   \\
& \bar{v}(x , z) := \sum_{j=1}^K \Pr\{z'|z\} \cdot v\left(x,z' \right)
\end{align}
$$

in which we split the problem into two equations: one Bellman equation, and one expectation equation.
This formulation is ready for applying the collocation.


## Collocation: Bellman equation

Consider a discretization strategy that discretize the state space of $x$ into $N$ grid node points:

$$
\mathbf{X} := [x^1, x^2, \dots, x^N]' \in \mathbb{R}^{N \times D}
$$

The above Bellman equation and expectation equation must hold on all $x$ and $z$ grid nodes.

> **Remark**: Instead of stacking the joint space by column-wise Cartesian product $\mathbf{X}$ with $\mathbf{Z}$ directly, we will explicitly repeat the $N$-equations for $K$ times for the convenience of expectation later.


In general, we can always write an interpolant of a scalar function $f(x):\mathbb{R}^D \to \mathbb{R}$ as:

$$
\hat{f}(x) := \phi(x) \cdot \theta
$$

where $P-1$ is the degree of interpolation; $\phi(x):\mathbb{R}^D\to\mathbb{R}^{1\times P}$ is the basis function (row vector); $\theta \in\mathbb{R}^P$ is the interpolation coefficient vector.
Vectorizing it over $N$ points:

$$
\begin{align}
\Phi(\mathbf{X}) := \begin{bmatrix}
\phi(x^1) \\ \phi(x^2) \\ \vdots \\ \phi(x^N)
\end{bmatrix} \in\mathbb{R}^{N \times P}
\end{align}
$$

Then $\Phi(\mathbf{X}) \cdot \theta$ evaluates the interpolant at all the $N$ points collectively.


Now, let's collocate the Bellman equation.
Define the following interpolations for each single $z$ state:

$$
\begin{align}
& v(x,z) \approx \hat{v}(x|z) := \phi(x|z) \cdot \theta(z) ; \theta(z) \in\mathbb{R}^P  \\
& \bar{v}(x,z) \approx \hat{\bar{v}}(x|z) := \phi_E(x|z) \cdot \theta_E (z); \theta_E(z)\in\mathbb{R}^{P_E}
\end{align}
$$

> **Remark**: Typically $P = P_E = N$ holds for interpolation. But we still distinguish them for generalizability in case of function approximations.

> **Remark**: If $P = P_E = N$ holds, then this brings extra convenience in stacking the equations.

We then stacking the Bellman equation at all $N$ nodes of $x$ and all $K$ nodes of $z$:

$$
\begin{aligned}
& \Phi(\mathbf{X}|z^1) \cdot \theta(z^1) = \mathbf{U}(\mathbf{X},z^1) + \beta \cdot \Phi\left( \mathfrak{X}(\mathbf{X},z^1) | z^1 \right) \cdot \theta_E (z^1)  \\
& \Phi(\mathbf{X}|z^2) \cdot \theta(z^2) = \mathbf{U}(\mathbf{X},z^2) + \beta \cdot \Phi\left( \mathfrak{X}(\mathbf{X},z^2) | z^2 \right) \cdot \theta_E (z^2)  \\
& \vdots \\
& \Phi(\mathbf{X}|z^K) \cdot \theta(z^K) = \mathbf{U}(\mathbf{X},z^K) + \beta \cdot \Phi\left( \mathfrak{X}(\mathbf{X},z^K) | z^K \right) \cdot \theta_E (z^K)  
\end{aligned}
$$

where there are total $NK$ pesudo linear equations and:

- $\mathbf{U}(\mathbf{X},z^k) := [u(x^1,z^k),u(x^2,z^k),\dots,u(x^N,z^k)]' \in\mathbb{R}^{N}$ is the stacking of the (optimized) instantaneous utilities conditional on today's shock realization being $z^i$.
- $\mathfrak{X}(\mathbf{X},z^k) := [\mathfrak{X}(x^1,z^k),\mathfrak{X}(x^2,z^k),\dots,\mathfrak{X}(x^N,z^k)]' \in\mathbb{R}^{N\times D} $ is the stacking of the state equations of the endogeneous states, conditional on today's shock realization being $z^k$. There is some degree of notation abuse but it should be harmless.


Let's further stack the system by defining:

$$
\begin{align}
& \vec\Phi (\mathbf{X}) := \begin{bmatrix}
\Phi(\mathbf{X}|z^1)  &   & \\
& \ddots  & \\
&& \Phi(\mathbf{X}|z^K)  
\end{bmatrix}   \in\mathbb{R}^{NK \times KP}  \\
% -------
& \vec\theta := \begin{bmatrix}
\theta(z^1)  \\ \vdots   \\ \theta(z^K)
\end{bmatrix} \in\mathbb{R}^{KP \times 1}    \\
% ===============
& (\vec\Phi_E \circ \mathfrak{X})(\mathbf{X}) := \begin{bmatrix}
\Phi_E(\mathfrak{X}(\mathbf{X},z^1)|z^1)  &   & \\
& \ddots  & \\
&& \Phi_E(\mathfrak{X}(\mathbf{X},z^K)|z^K)
\end{bmatrix}   \in\mathbb{R}^{NK \times KP_E}  \\
% -------
& \vec\theta_E := \begin{bmatrix}
\theta_E(z^1)  \\ \vdots   \\ \theta_E(z^K)
\end{bmatrix} \in\mathbb{R}^{KP_E \times 1}  \\
% =============
& \vec{\mathbf{U}}(\mathbf{X}) := \begin{bmatrix}
    \mathbf{U}(\mathbf{X},z^1) \\
    \vdots \\
    \mathbf{U}(\mathbf{X},z^K)
\end{bmatrix}  \in\mathbb{R}^{NK \times 1}
\end{align}
$$

Then, the Bellman equation is finally stacked as:

$$
\vec\Phi(\mathbf{X}) \cdot \vec\theta = \vec{\mathbf{U}}(\mathbf{X}) + \beta \cdot (\vec\Phi_E \circ \mathfrak{X})(\mathbf{X}) \cdot \vec\theta_E 
$$

> **Remark**: The final stacking system looks the same as directly Cartesian producting everything of $x$ and $z$. However, 1. the intermediate construction provides great convenience of notation; 2. allows $\mathbf{X}$ to be not necessarily created by Cartesian product (only assumes the joining between $x$ and $z$ is Cartesian); 3. avoids assuming $z$ is continuous to improve the sparsity of the system.

> **Remark**: In fact, with the guess of $\theta,\theta_E$ from previous iterations of the fixed point iteration, the system is block-solvable and there is no need to stack everything and solve a so big system. However, if using gradient-based methods, then all equations must be solved jointly.


## Collocation: Expectation equation

Due to the Markovian property of the $z$ process, it is non-trivial to collocate the expectation equations.
Conditional on shock state $z^k$, the expectation equation can be linearized as:

$$
\begin{aligned}
& \bar{v}(x,z^k) = \mathbf{P}^z_k \cdot \begin{bmatrix}
v(x,z^1) \\
\vdots \\
v(x,z^K)
\end{bmatrix} \\
% ------
\implies & \underbrace{\phi_E(x|z^k)}_{\in\mathbb{R}^{1\times P_E}} \cdot \theta_E (z^k) = \underbrace{ \mathbf{P}^z_k }_{\in\mathbb{R}^{1\times K}} \cdot \underbrace{ \begin{bmatrix}
\phi(x|z^1) \\
& \ddots \\
&& \phi(x|z^K)
\end{bmatrix}  }_{\in\mathbb{R}^{K\times KP}} \cdot \underbrace{ \begin{bmatrix}
\theta (z^1) \\
\vdots \\
\theta (z^K)
\end{bmatrix}  }_{\in\mathbb{R}^{KP \times 1}}  \\
% ------
& \phi_E(x|z^k) \cdot \theta_E (z^k) = \mathbf{P}^z_k \cdot \tilde\Phi(x|\mathbf{Z})  \cdot \vec\theta
\end{aligned}
$$

where $\tilde\Phi(x|\mathbf{Z}) := \text{rdiagm}\{ \Phi(x|\mathbf{Z}) \}$ creates a block diagonal matrix by using every row of the basis matrix $\Phi(x|\mathbf{Z}) \in\mathbb{R}^{K\times P} $ as each diagonal block.



To be consistent with the stacking Bellman equations, we consider:

$$
\begin{aligned}
\underbrace{  \Phi_E(\mathbf{X}|z^k)  }_{\in\mathbb{R}^{N\times P_E}} \cdot \theta_E =& \underbrace{ \begin{bmatrix}
    \mathbf{P}^z_k \\
    & \ddots \\
    && \mathbf{P}^z_k
\end{bmatrix} }_{\in\mathbb{R}^{N\times NK }} \cdot \underbrace{\begin{bmatrix}
    \tilde\Phi(x^1|\mathbf{Z}) \\
    \vdots \\
    \tilde\Phi(x^N|\mathbf{Z})
\end{bmatrix}}_{\in\mathbb{R}^{NK\times KP}} \cdot \underbrace{ \begin{bmatrix}
    \theta (z^1) \\
    \vdots \\
    \theta (z^K)
\end{bmatrix}  }_{\in\mathbb{R}^{KP \times 1}}  \\
% -----
=& (I_N \otimes \mathbf{P}^z_k) \cdot \underbrace{\begin{bmatrix}
    \tilde\Phi(x^1|\mathbf{Z}) \\
    \vdots \\
    \tilde\Phi(x^N|\mathbf{Z})
\end{bmatrix}}_{\in\mathbb{R}^{NK\times KP}} \cdot \underbrace{ \begin{bmatrix}
    \theta (z^1) \\
    \vdots \\
    \theta (z^K)
\end{bmatrix}  }_{\in\mathbb{R}^{KP \times 1}}
\end{aligned}
$$


One can quickly realize that the 2nd term of the right hand side is a **_row_** permutation of $\vec\Phi(\mathbf{X}) \in \mathbb{R}^{NK \times KP}$. The 2nd term and the 3rd term are invariant to all $z^k$.

If we stack all $K$ groups of expectation equations:

$$
\begin{aligned}
\underbrace{ \vec\Phi_E (\mathbf{X}) }_{\in\mathbb{R}^{NK\times KP_E}} \cdot \vec\theta_E  =&  \underbrace{ \vec{\mathbf{P}}^z }_{\in\mathbb{R}^{NK\times NK}} \cdot \underbrace{ \vec{\tilde{\Phi}}(\mathbf{X}) }_{\in\mathbb{R}^{NK\times KP}} \cdot \vec\theta  \\
% -----
=& \vec{\mathbf{P}}^z \cdot M \cdot  \vec\Phi(\mathbf{X}) \cdot  \vec\theta  
\end{aligned}
$$

where 

$$
\vec{\mathbf{P}}^z := \begin{bmatrix}
    I_N \otimes \mathbf{P}^z_1 \\
    \vdots \\
    I_N \otimes \mathbf{P}^z_K
\end{bmatrix} \in\mathbb{R}^{NK\times NK}
$$

> **Remark**: $\vec{\mathbf{P}}^z = $ `vcat([kron(I(N),prow') for prow in P |> eachrow]...)`, or sparse version:
> ```julia
> vcat([ blockdiag([ prow' |> sparse for _ in 1:N ]...) for prow in P |> eachrow]...)
> ```


and, $M_1 \in \mathbb{R}^{NK\times NK}$ is a row permutation matrix that permutes

$$
\begin{bmatrix}
\phi(x^1,z^1) \\
\phi(x^2,z^1) \\
\vdots \\
\phi(x^N,z^1) \\
& \phi(x^1,z^2) \\
& \vdots \\
& \phi(x^N,z^2) \\
&& \ddots \\
&&& \phi(x^1,z^K) \\
&&& \vdots \\
&&& \phi(x^N,z^K)
\end{bmatrix}
$$

to

$$
\begin{bmatrix}
\phi(x^1,z^1) \\
& \phi(x^1,z^2) \\
&& \ddots \\
&&& \phi(x^1,z^K) \\
\phi(x^2,z^1) \\
& \phi(x^2,z^2) \\
&& \ddots \\
&&& \phi(x^2,z^K) \\
\vdots & \vdots  & \dots & \vdots \\
\phi(x^N,z^1) \\
& \phi(x^N,z^2) \\
&& \ddots \\
&&& \phi(x^N,z^K) \\
\end{bmatrix}
$$


One can quickly realize that the most difficult part of this section is to build $M$.

> **Remark**: This permutation is equivalent to switching the order of Cartesian product between $x$'s space and $z$' space.



## Special case

The above is the very verbose version of the collocation which requires creating many big (but sparse) matrices.
However, the following easy-to-satisfy assumptions greatly simplifies the problem:

1. $\phi (x|z) = \phi_E (x|z) = \phi(x)$, i.e. the basis function is independent from the shock $z$ and uniform across all places that need to interpolate a function mapping from $\mathbb{R}^{D}$
2. $P = P_E = N$, i.e. use interpolation rather than approximation, and use the same interpolation method for both value function and the expected value function

all the $K$ groups of stacking equations collapse to $2 K$ coefficient vectors $(\theta(z^k),\theta_E(z^k))$. The basis matrices at the grid nodes $\Phi(\mathbf{X})$ can be pre-conditioned and used everywhere.



Therefore, for shock state $z^k$, there are $2N$ pseudo-linear equations:

$$
\begin{aligned}
& \Phi(\mathbf{X}) \cdot \theta(z^k) = \mathbf{U}(\mathbf{X}) + \beta \cdot \Phi(\mathfrak{X}(\mathbf{X},z^k)) \cdot \theta_E (z^k)  \\ 
& \Phi(\mathbf{X}) \cdot \theta_E(z^k) = (I_{N} \otimes \mathbf{P}^z_k) \cdot M \cdot \vec\Phi(\mathbf{X}) \cdot \vec\theta
\end{aligned}
$$

in which we can see that the expectation equations simultaneously depends on all the Bellman equation stackings of the total $K$ shock states.

Does this mean that we have to jointly solve all $2NK$ pseudo-linear equations? The answer is not necessary: If one is doing fixed point iteration on $\vec\theta$, then in each iteration, one can independently update $\theta(z^k)$ for each $z^k$ block of Bellman equations taking the last iteration's $\theta_E(z^k)$ as given. After solving all blocks, she can update the guess of the stacking $\vec\theta$, then update each $\theta_E(z^k)$ by solving the linear equations of the expectation equations blockwise and then enter the next iteration.

However, if one wants to accelerate the computation using Newton-like methods, then all $2NK$ equations must be fully stacked and updated jointly:

$$
\begin{align}
& \vec\Phi(\mathbf{X}) \cdot \vec\theta = \vec{\mathbf{U}} (\mathbf{X})  + \beta \cdot \vec\Phi\left( \vec{\mathfrak{X}}(\mathbf{X}) \right) \cdot \vec\theta_E    \\
& \vec\Phi(\mathbf{X}) \cdot \vec\theta_E = \vec{\mathbf{P}}^z \cdot M \cdot \vec\Phi(\mathbf{X}) \cdot \vec\theta  \nonumber
\end{align}
$$

Or formally,


$$
\begin{align}
    \begin{bmatrix}
        \vec\Phi(\mathbf{X}) & \mathbf{0}_{NK\times NK} \\
        \mathbf{0}_{NK\times NK} & \vec\Phi(\mathbf{X})
    \end{bmatrix} \cdot \begin{bmatrix}
        \vec\theta \\
        \vec\theta_E
    \end{bmatrix} = \begin{bmatrix}
        \vec{\mathbf{U}} (\mathbf{X}) \\
        \mathbf{0}_{NK\times 1}
    \end{bmatrix} + \begin{bmatrix}
        \beta \cdot \vec\Phi\left( \vec{\mathfrak{X}}(\mathbf{X}) \right) & \mathbf{0}_{NK\times NK} \\
        \mathbf{0}_{NK\times NK} & \vec{\mathbf{P}}^z \cdot M \cdot \vec\Phi(\mathbf{X}) 
    \end{bmatrix} \cdot \begin{bmatrix}
        \vec\theta_E \\
        \vec\theta
    \end{bmatrix}
\end{align}
$$

> **Remark**: Thanks to the separation of endogenous states $x$ and shock $z$, the formal fully stacked system has regular blank areas $\mathbf{0}_{NK\times NK}$ which is much clear and independent from $J$ which is the dimensionality of $z$.

## Jacobian & Newton's method

Let's stick to the special case.
Thanks to the pseudo linearity, it is possible to jointly solve the coefficients $\bm{\theta} := (\vec\theta,\vec\theta_E) \in\mathbb{R}^{2NK} $ from the fully stacked system using gradient-based methods to accelerate the convergence.
In this case, providing a user-specified Jacobian can greatly accelerate the computation.

Let's denote the fully stacked system as a single function:

$$
\mathbf{F}(\bm{\theta}) := RHS - LHS = \mathbf{0}_{2NK\times 1}
$$

where RHS is the right hand side of the fomal fully stacked system; and LHS is the left hand side of that system.
Then, the Jacobian is:

$$
    \mathbf{J}(\mathbf{X}) := \begin{bmatrix}
        \mathbf{0}_{NK\times NK} & \beta \cdot \vec\Phi\left( \vec{\mathfrak{X}}(\mathbf{X}) \right) \\
        \vec{\mathbf{P}}^z \cdot M \cdot \vec\Phi(\mathbf{X})  & \mathbf{0}_{NK\times NK} 
    \end{bmatrix} - \begin{bmatrix}
        \vec\Phi(\mathbf{X}) & \mathbf{0}_{NK\times NK} \\
        \mathbf{0}_{NK\times NK} & \vec\Phi(\mathbf{X})
    \end{bmatrix}
$$

Apparently, $\mathbf{J}(\mathbf{X})$ is non-singular if $\vec\Phi(\mathbf{X})$ is non-singular. This property is satisfied in many interpolation methods such as piecewise linear interpolations.


As a reminder, the vanilla Newton's method formula is:

$$
\bm\theta^{j+1} = \bm\theta^{j} - \mathbf{J}^{-1}(\mathbf{X})  \cdot \mathbf{F}(\bm\theta)
$$





## Performance bottleneck

There are several bottlenecks of the performance:

1. The optimization step which solves the optimal policy $\mathfrak{C}(\cdot)$ and evalutes $\mathbf{U}(\mathbf{X})$. This takes the longest time of the whole process.
2. Evaluating $\Phi(\mathfrak{X}(\mathbf{X},z^k))$ basis matrix which changes in each iteration because the policy function is updated in each iteration. This basis matrix cannot be pre-conditioned.
3. Evaluating the Jacobian of the fully stacked system also needs to evalaute $\vec\Phi\left( \vec{\mathfrak{X}}(\mathbf{X}) \right)$, which may waste time and memory if the solver cannot recycle the intermediate results.














