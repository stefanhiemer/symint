Examples
========

The ``examples/`` directory of the repository contains runnable
scripts:

``01_gaussian_moments.py``
   Gaussian moments on the whole line, a half-line and :math:`[0, 1]`,
   with the built-in rule only.
``02_fem_element_matrices.py``
   P1 and P2 mass matrices on the reference triangle, and NumPy code
   with an array-valued coefficient (:doc:`codegen`).
``03_coulomb_gaussians.py``
   The Coulomb integral of two Gaussians, centred and shifted
   (:doc:`transforms`).
``04_partial_results.py``
   A partial result evaluated by generated quadrature code.
``05_custom_rule.py``
   A user rule for Slater-type moments (:doc:`rules`).
``06_verify.py``
   Checking antiderivatives and definite results (:doc:`verification`).
``07_qc_azimuthal.py``
   The quantum-chemistry rules (:doc:`contexts`).

Gaussian moments
----------------

.. literalinclude:: ../../examples/01_gaussian_moments.py
   :language: python
