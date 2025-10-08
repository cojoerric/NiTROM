Cavity Flow
=========

We use a lid-driven cavity flow model shown in :cite:`padovan2024data` as one 
practical application of the model reduction techniques implemented in 
:code:`NiTROM`. The flow dynamics are governed by the incompressible 
Navier-Stokes equations and the continuity equation:

    .. math::

        \begin{align}
            \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla 
            \mathbf{v} &= -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{v},\\
            \nabla \cdot \mathbf{v} &= 0,
        \end{align}

where :math:`\mathbf{v}(\mathbf{x},t)=(u(\mathbf{x},t), v(\mathbf{x},t))` is the 
2D velocity field, :math:`p(\mathbf{x},t)` is the pressure field, and :math:`Re` 
is the Reynolds number. The 2D spatial domain is a 2D square cavity 
:math:`D = [0,1]\times [0,1]` with no-slip boundary conditions on all walls except 
for the upper wall, where we impose a unit tangential velocity :math:`u=1`.


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