Non-separable Integrands and Useful Transforms
==============================================

Separation (:doc:`pipeline`) only helps if the integrand is a product
of factors that each depend on different variables, thus a single mixed 
factor prevents it, e.g. in

.. math::

   \int\!\!\int \frac{e^{-x^2 - y^2}}{\sqrt{1 + (x - y)^2}}\,dx\,dy

the Gaussians would separate into an x-part and a y-part, but the root
stops this.

Such factors are common and often have the form :math:`Q^{-\nu/2}`, with
:math:`Q` a quadratic polynomial in the integration variables
(:math:`Q = 1 + (x - y)^2`, :math:`\nu = 1` above).

``transform=`` rewrites such a factor as an integral over a new outer
variable :math:`t`, in which the variables appear only in an
exponential :math:`e^{-t^2 Q}`. For fixed :math:`t` this is a Gaussian
in the original variables, and the Gaussians separate. The available 
representations:

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
last. With ``substitute_outer=True`` (default) the Gaussian and Laplace
forms are first mapped onto :math:`[0, 1]` by
:math:`u^2 = t^2/(p + t^2)`.

Example: the electrostatic interaction of two Gaussian charge
distributions, where :math:`1/|r_1 - r_2|` mixes all six coordinates.

.. literalinclude:: ../../examples/03_coulomb_gaussians.py
   :language: python

Terms without a coupling factor are left as they are. An unknown
``transform`` raises ``ValueError``; the valid names are in
`~symintegral.rewrite.TRANSFORMS`.
