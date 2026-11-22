Coupled integrands: transforms
==============================

Separation needs factors that share no variables. A factor like
:math:`1/|r_1 - r_2|` couples all six coordinates, so it would block
separation of the whole integral. ``transform=`` replaces factors
:math:`Q^{-\nu/2}`, :math:`Q` quadratic in the integration variables, by
an integral representation over a new outer variable :math:`t`:

``"gaussian"``
   .. math::

      Q^{-\nu/2} = \frac{2}{\Gamma(\nu/2)} \int_0^\infty t^{\nu-1}
      e^{-t^2 Q}\,dt

``"laplace"``
   The same with :math:`s = t^2`.

``"fourier"``
   .. math::

      \frac{1}{|l|} = \frac{1}{2\pi^2} \int \frac{e^{i k\cdot l}}{k^2}\,d^3k

After the transform the original variables appear only in Gaussians, so
separation and the Gaussian rule apply. The outer variable is integrated
last; with ``substitute_outer=True`` (default) the Gaussian and Laplace
forms are first mapped onto :math:`[0, 1]` by
:math:`u^2 = t^2/(p + t^2)`.

Example: the Coulomb interaction of two Gaussian charge distributions.

.. literalinclude:: ../../examples/03_coulomb_gaussians.py
   :language: python

Terms without a coupling factor are left as they are. An unknown
``transform`` raises ``ValueError``; the valid names are in
`~symintegral.rewrite.TRANSFORMS`.
