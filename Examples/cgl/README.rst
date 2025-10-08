Complex Ginzburg-Landau Equation
=========

We consider the complex Ginzburg-Landau (CGL) equation as a canonical model for
nonlinear dynamics, as described in :cite:`padovan2024data`. The CGL equation 
is given by

.. math::
   \begin{equation}
   \frac{\partial q}{\partial t} =
   \left( -\nu \frac{\partial}{\partial x} +
   \gamma \frac{\partial^2}{\partial x^2} + \mu \right) q
   - a |q|^2 q,
   \end{equation}

where $x \in (-\infty, \infty)$, $q(x,t) \in \mathbb{C}$, and the parameters are
$a = 0.1$, $\gamma = 1 - i$, $\nu = 2 + 0.4i$,
and $\mu = (\mu_0 - 0.2^2) + \mu_2 \frac{x^2}{2}$,
with $\mu_2 = -0.1$ and $\mu_0 = 0.38$.

We are interested in creating a ROM to predict the time history of the complex-valued
measurements :math:`y` in response to the complex-valued input dynamics :math:`Bu`:

.. math::

   y = Cq = \\exp\\left\\{ -\\left( \\frac{x + \\bar{x}}{s} \\right)^2 \\right\\} q \\\\
   Bu = \\exp\\left\\{ -\\left( \\frac{x - \\bar{x}}{s} \\right)^2 \\right\\} u

After spatial discretization, the equation above is cast into a real-valued
dynamical system with cubic dynamics:

.. math::

   \\frac{d\\mathbf{q}}{dt} = \\mathbf{Aq + H : (q \\otimes q \\otimes q) + Bu} \\\\
   \\mathbf{y} = \\mathbf{Cq}


Instructions
------------

1. Collect training trajectories by running

   .. code-block:: bash

      mpiexec -n 2 python -u 




2. Apply model reduction and reconstruct the flow field using

   .. code-block:: bash

      mpiexec -n 2 python -u 


3. Navigate to the :file:`results/` directory to check out the results.

Scripts
-------
