import torch
import math
import warnings

#todo. migration step5 or keep import
from legacy.neuralop.losses.differentiation import FiniteDiff

warnings.filterwarnings("once", category=UserWarning)


class LpLoss(object):
    """LpLoss provides the Lp norm between two discretized d-dimensional functions.

    Note that LpLoss always averages over the spatial dimensions.

    .. note ::
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular
        discretization and take ``1 / measure`` as the quadrature weights.

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``LpLoss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``.
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning::

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss
    """

    def __init__(self, d=1, p=2, measure=1.0, reduction="sum", eps=1e-8):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps

        allowed_reductions = ["sum", "mean"]
        assert (
                reduction in allowed_reductions
        ), f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure] * self.d
        else:
            self.measure = measure

    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"

    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that
        ``LpLoss`` computes the average over spatial dims.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0] * self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j] / x.size(-j)

        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == "sum":
            x = torch.sum(x)
        else:
            x = torch.mean(x)

        return x

    def abs(self, x, y, quadrature=None, take_root=True):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        # Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        diff_flat = torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d)

        if self.p == 1:
            const = math.prod(quadrature)
            diff = const * torch.sum(torch.abs(diff_flat), dim=-1, keepdim=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            const = math.prod(quadrature)
            diff = const * torch.sum(diff_flat ** self.p, dim=-1, keepdim=False)
        else:
            const = math.prod(quadrature)
            diff = const * torch.sum(
                torch.abs(diff_flat) ** self.p, dim=-1, keepdim=False
            )

        if take_root and self.p != 1:
            diff = diff ** (1.0 / self.p)

        diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, take_root=True):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """

        diff_flat = torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d)
        y_flat = torch.flatten(y, start_dim=-self.d)

        if self.p == 1:
            diff = torch.sum(torch.abs(diff_flat), dim=-1, keepdim=False)
            ynorm = torch.sum(torch.abs(y_flat), dim=-1, keepdim=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            diff = torch.sum(diff_flat ** self.p, dim=-1, keepdim=False)
            ynorm = torch.sum(y_flat ** self.p, dim=-1, keepdim=False)
        else:
            diff = torch.sum(torch.abs(diff_flat) ** self.p, dim=-1, keepdim=False)
            ynorm = torch.sum(torch.abs(y_flat) ** self.p, dim=-1, keepdim=False)

        if take_root and self.p != 1:
            diff = (diff ** (1.0 / self.p)) / (ynorm ** (1.0 / self.p) + self.eps)
        else:
            diff = diff / (ynorm + self.eps)

        diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, **kwargs):
        if kwargs:
            warnings.warn(
                f"LpLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2
            )
        return self.rel(y_pred, y)


class H1Loss(object):
    """H1 Sobolev norm between two d-dimensional discretized functions.

    .. note ::
        In function space, the Sobolev norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to H1Loss, we assume a regular
        discretization and take ``1 / measure`` as the quadrature weights.

    Parameters
    ----------
    d : int, optional
        dimension of input functions, by default 1
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``H1Loss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``.

    reduction : str, optional
        whether to reduce across the batch and channel dimension
        by summing ('sum') or averaging ('mean')

        .. warning :

            H1Loss always averages over the spatial dimensions.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss
    periodic_in_x : bool, optional
        whether to use periodic boundary conditions in x-direction when computing finite differences:
        - True: periodic in x (default)
        - False: non-periodic in x with forward/backward differences at boundaries
        by default True
    periodic_in_y : bool, optional
        whether to use periodic boundary conditions in y-direction when computing finite differences:
        - True: periodic in y (default)
        - False: non-periodic in y with forward/backward differences at boundaries
        by default True
    """

    def __init__(
            self,
            d=1,
            measure=1.0,
            reduction="sum",
            eps=1e-8,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True,
    ):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z

        self.eps = eps

        allowed_reductions = ["sum", "mean"]
        assert (
                reduction in allowed_reductions
        ), f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure] * self.d
        else:
            self.measure = measure

    @property
    def name(self):
        return f"H1_{self.d}DLoss"

    def compute_terms(self, x, y, quadrature):
        """compute_terms computes the necessary
        finite-difference derivative terms for computing
        the H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : int or list
            quadrature weights

        """
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            fd1d = FiniteDiff(dim=1, h=quadrature[0], periodic_in_x=self.periodic_in_x)
            x_x = fd1d.dx(x)
            y_x = fd1d.dx(y)

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            fd2d = FiniteDiff(dim=2, h=quadrature, periodic_in_x=self.periodic_in_x, periodic_in_y=self.periodic_in_y)
            x_x, x_y = fd2d.dx(x), fd2d.dy(x)
            y_x, y_y = fd2d.dx(y), fd2d.dy(y)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)

        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            fd3d = FiniteDiff(dim=3, h=quadrature, periodic_in_x=self.periodic_in_x, periodic_in_y=self.periodic_in_y,
                              periodic_in_z=self.periodic_in_z)
            x_x, x_y, x_z = fd3d.dx(x), fd3d.dy(x), fd3d.dz(x)
            y_x, y_y, y_z = fd3d.dx(y), fd3d.dy(y), fd3d.dz(y)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)

        return dict_x, dict_y

    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that
        ``LpLoss`` computes the average over spatial dims.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0] * self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j] / x.size(-j)

        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == "sum":
            x = torch.sum(x)
        else:
            x = torch.mean(x)

        return x

    def abs(self, x, y, quadrature=None, take_root=True):
        """absolute H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True
        """
        # Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const * torch.sum((dict_x[0] - dict_y[0]) ** 2, dim=-1, keepdim=False)

        for j in range(1, self.d + 1):
            diff += const * torch.sum((dict_x[j] - dict_y[j]) ** 2, dim=-1, keepdim=False)

        if take_root:
            diff = diff ** 0.5

        diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, quadrature=None, take_root=True):
        """relative H1-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True
        """
        # Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.sum((dict_x[0] - dict_y[0]) ** 2, dim=-1, keepdim=False)
        ynorm = torch.sum(dict_y[0] ** 2, dim=-1, keepdim=False)

        for j in range(1, self.d + 1):
            diff += torch.sum((dict_x[j] - dict_y[j]) ** 2, dim=-1, keepdim=False)
            ynorm += torch.sum(dict_y[j] ** 2, dim=-1, keepdim=False)

        if take_root:
            diff = (diff ** 0.5) / (ynorm ** 0.5 + self.eps)
        else:
            diff = diff / (ynorm + self.eps)

        diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, quadrature=None, take_root=True, **kwargs):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            normalization constant for reduction, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True
        """
        if kwargs:
            warnings.warn(
                f"H1Loss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2
            )
        return self.rel(y_pred, y, quadrature=quadrature)
