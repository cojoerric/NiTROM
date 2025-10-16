:orphan:

Cavity Flow
=========

We use a lid-driven cavity flow model shown in :cite:`padovan2024data` as one 
practical application of the model reduction techniques implemented in 
:code:`NiTROM`. The flow dynamics are governed by the incompressible 
Navier-Stokes equations and the continuity equation:

    .. math::

            \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla 
            \mathbf{v} &= -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{v},\\
            \nabla \cdot \mathbf{v} &= 0,

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



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_post_process_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_post_process.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Dec 29 19:21:27 2022</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_compute_baseflow_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_compute_baseflow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Tue Feb 21 23:43:38 2023</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_time_steppers_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_time_steppers.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Dec 29 12:02:38 2022</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_train_opinf_and_podgal_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_train_opinf_and_podgal.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:57:05 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_linear_operators_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_linear_operators.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Tue Dec 20 16:57:06 2022</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_test_new_cfd_code_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_test_new_cfd_code.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Wed Jun 19 15:40:38 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_run_optimization_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_run_optimization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:57:05 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_train_nitrom_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_train_nitrom.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:57:05 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_generate_data_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_generate_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:57:05 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_classes_cavity_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_classes_cavity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Tue Dec 20 15:38:17 2022</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_numba_operators_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_numba_operators.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Tue Dec 20 18:34:37 2022</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_generate_figures_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_generate_figures.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:00:43 2024</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="@author: alberto">

.. only:: html

  .. image:: /auto_examples/cavity_flow/images/thumb/sphx_glr_read_results_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_cavity_flow_read_results.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Created on Thu Jun 20 19:57:05 2024</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/cavity_flow/post_process
   /auto_examples/cavity_flow/compute_baseflow
   /auto_examples/cavity_flow/time_steppers
   /auto_examples/cavity_flow/train_opinf_and_podgal
   /auto_examples/cavity_flow/linear_operators
   /auto_examples/cavity_flow/test_new_cfd_code
   /auto_examples/cavity_flow/run_optimization
   /auto_examples/cavity_flow/train_nitrom
   /auto_examples/cavity_flow/generate_data
   /auto_examples/cavity_flow/classes_cavity
   /auto_examples/cavity_flow/numba_operators
   /auto_examples/cavity_flow/generate_figures
   /auto_examples/cavity_flow/read_results


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: cavity_flow_python.zip </auto_examples/cavity_flow/cavity_flow_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: cavity_flow_jupyter.zip </auto_examples/cavity_flow/cavity_flow_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
