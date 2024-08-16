import heapq
import itertools

from dataclasses import dataclass
from scipy._lib._util import MapWrapper

import numpy as np

from scipy.integrate._rules import (
    ProductNestedFixed, NestedFixedRule,
    GaussKronrodQuadrature, NewtonCotesQuadrature,
    GenzMalikCubature,
)


@dataclass
class CubatureRegion:
    estimate: np.ndarray
    error: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def __lt__(self, other):
        # Consider regions with higher error estimates as being "less than" regions with
        # lower order estimates, so that regions with high error estimates are placed at
        # the top of the heap.
        return _max_norm(self.error) > _max_norm(other.error)


@dataclass
class CubatureResult:
    estimate: np.ndarray
    error: np.ndarray
    success: bool
    status: str
    regions: list[CubatureRegion]
    subdivisions: int
    atol: float
    rtol: float


def cubature(f, a=None, b=None, rule="gk21",
             rtol=1e-05, atol=1e-08,
             max_subdivisions=10000,
             args=(), kwargs=None,
             workers=1, points=None, tr=None,
             region=None):
    r"""
    Adaptive cubature of multidimensional array-valued function.

    Given an arbitrary integration rule, this function returns an estimate of the
    integral to the requested tolerance over the region defined by the arrays `a` and
    `b` specifying the corners of a hypercube.

    Convergence is not guaranteed for all integrals.

    Parameters
    ----------
    f : callable
        Function to integrate. `f` must have the signature::
            f(x : ndarray, \*args, \*\*kwargs) -> ndarray

        `f` should accept arrays ``x`` of shape::
            (npoints, ndim)

        and output arrays of shape::
            (npoints, output_dim_1, ..., output_dim_n)

        In this case, `cub` will return arrays of shape::
            (output_dim_1, ..., output_dim_n)
    a, b : array_like or float
        Lower and upper limits of integration as rank-1 arrays specifying the left and
        right endpoints of the intervals being integrated over. If a float is passed,
        these will be converted to singleton arrays. Infinite limits are currently not
        supported.
    rule : str, optional
        Rule used to estimate the integral. If passing a string, the options are
        "gauss-kronrod" (21 node), "newton-cotes" (5 node) or "genz-malik" (degree 7).
        If a rule like "gauss-kronrod" or "newton-cotes" is specified for an ``n``-dim
        integrand, the corresponding Cartesian product rule is used.

        "gk21", "gk15" and "trapezoid" are also supported for compatibility with
        `quad_vec`.
    rtol : float, optional
        Relative tolerance. Default is 1e-05.
    atol : float, optional
        Absolute tolerance. Default is 1e-08.
    max_subdivisions : int, optional
        Upper bound on the number of subdivisions to perform to improve the estimate
        over a subregion. Default is 10,000.
    args : tuple, optional
        Additional positional args passed to `f`, if any.
    kwargs : tuple, optional
        Additional keyword args passed to `f`, if any.
    workers : int or map-like callable, optional
        If `workers` is an integer, part of the computation is done in parallel
        subdivided to this many tasks (using :class:`python:multiprocessing.pool.Pool`).
        Supply `-1` to use all cores available to the Process. Alternatively, supply a
        map-like callable, such as :meth:`python:multiprocessing.pool.Pool.map` for
        evaluating the population in parallel. This evaluation is carried out as
        ``workers(func, iterable)``.

    Returns
    -------
    res : CubatureResult
        Result of estimation. See `CubatureResult`.

    Examples
    --------
    A simple 1D integral with vector output. Here ``f(x) = x^n`` is integrated over the
    interval ``[0, 1]``. Since no rule is specified, the default "gk21" is used, which
    corresponds to `GaussKronrod` rule with 21 nodes.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> def f(x, n):
    ...    return x.reshape(-1, 1)**n  # Make sure x and n are broadcastable
    >>> res = cubature(
    ...     f,
    ...     a=[0],
    ...     b=[1],
    ...     args=(
    ...         # Since f accepts arrays of shape (npoints, ndim) we need to
    ...         # make sure n is the right shape
    ...         np.arange(10).reshape(1, -1),
    ...     )
    ... )
    >>> res.estimate
     array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ,
            0.16666667, 0.14285714, 0.125     , 0.11111111, 0.1       ])

    A 7D integral with arbitrary-shaped array output. Here::
        f(x) = cos(2*pi*r + alphas @ x)

    for some ``r`` and ``alphas``, and the integral is performed over the unit
    hybercube, :math:`[0, 1]^7`. Since the integral is in a moderate number of
    dimensions, "genz-malik" is used rather than the default "gauss-kronrod" to avoid
    constructing a product rule with :math:`21^7 \approx 2 \times 10^9` nodes.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> def f(x, r, alphas):
    ...     # f(x) = cos(2*pi*r + alpha @ x)
    ...     # Need to allow r and alphas to be arbitrary shape
    ...     npoints, ndim = x.shape[0], x.shape[-1]
    ...     alphas_reshaped = alphas[np.newaxis, :]
    ...     x_reshaped = x.reshape(npoints, *([1]*(len(alphas.shape) - 1)), ndim)
    ...     return np.cos(2*np.pi*r + np.sum(alphas_reshaped * x_reshaped, axis=-1))
    >>> np.random.seed(1)
    >>> r, alphas = np.random.rand(2, 3), np.random.rand(2, 3, 7)
    >>> res = cubature(
    ...     f=f,
    ...     a=np.array([0, 0, 0, 0, 0, 0, 0]),
    ...     b=np.array([1, 1, 1, 1, 1, 1, 1]),
    ...     rule="genz-malik",
    ...     kwargs={
    ...         "r": r,
    ...         "alphas": alphas,
    ...     }
    ... )
    >>> res.estimate
     array([[-0.61336595,  0.88388877, -0.57997549],
            [-0.86968418, -0.86877137, -0.64602074]])

    To compute in parallel, it is possible to use the argument `workers`, for example:

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> with ThreadPoolExecutor() as executor:
    ...     res = cubature(
    ...         f=f,
    ...         a=np.array([0, 0, 0, 0, 0, 0, 0]),
    ...         b=np.array([1, 1, 1, 1, 1, 1, 1]),
    ...         rule="genz-malik",
    ...         kwargs={
    ...             "r": r,
    ...             "alphas": alphas,
    ...         },
    ...         workers=executor.map,
    ...      )
    >>> res.estimate
     array([[-0.61336595,  0.88388877, -0.57997549],
            [-0.86968418, -0.86877137, -0.64602074]])

    When this is done with process-based parallelization (as would be the case passing
    `workers` as an integer, you should ensure the main module is import-safe.

    Notes
    -----

    If passing a list of `points`, the corresponding rule should not evaluate `f` at the
    boundary of a region. For this reason, it's not possible to use `points` and
    ``NewtonCotes()`` at the same time.

    If using process-based parallelization, as is the case when `workers` is an integer
    not equal to 1, or using `multiprocessing.Pool.map`, the main module needs to be
    import-safe.
    """

    # It is also possible to use a custom rule, but this is not yet part of the public
    # API. An example of this can be found in the class scipy.integrate._rules.Rule.

    kwargs = kwargs or {}
    max_subdivisions = np.inf if max_subdivisions is None else max_subdivisions
    points = [] if points is None else points

    # Convert a and b to arrays of at least 1D
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b should be 1D arrays")

    if region is not None:
        # TODO: report error if incompatible options
        f = FuncLimitsTransform(f, a, b, region)
        a, b = f.limits

    # Apply a transformation if one was specified
    if tr is not None:
        f = tr(f, a, b)
        a, b = f.limits

    # If any limits are the wrong way around, flip them and keep track of the sign.
    sign = (-1) ** np.sum(a > b)  # false = 0, true = 1
    a_flipped = np.min(np.array([a, b]), axis=0)
    b_flipped = np.max(np.array([a, b]), axis=0)

    # If necessary, apply a transformation to handle infinite limits
    if np.any(np.isinf(a_flipped)) or np.any(np.isinf(b_flipped)):
        f = InfiniteLimitsTransform(f, a_flipped, b_flipped)
        a, b = f.limits
        points.extend(f.points)
    else:
        f = f
        a, b = a_flipped, b_flipped

    # If any problematic points are specified, divide the initial region so that
    # these points lie on the edge of a subregion, which means ``f`` won't be evaluated
    # there (if the rule being used has no evaluation points on the boundary).
    if points == []:
        initial_regions = [(a, b)]
    else:
        initial_regions = _split_at_points(a, b, points)

    # If the rule is a string, convert to a corresponding rule
    if isinstance(rule, str):
        ndim = len(a)

        if rule == "genz-malik":
            rule = GenzMalikCubature(ndim)
        else:
            quadratues = {
                "gauss-kronrod": GaussKronrodQuadrature(21),
                "newton-cotes": NestedFixedRule(
                    NewtonCotesQuadrature(5),
                    NewtonCotesQuadrature(3),
                ),

                # Also allow names quad_vec uses:
                "gk21": GaussKronrodQuadrature(21),
                "gk15": GaussKronrodQuadrature(15),
                "trapezoid": NestedFixedRule(
                    NewtonCotesQuadrature(5),
                    NewtonCotesQuadrature(3),
                ),
            }

            base_rule = quadratues.get(rule)

            if base_rule is None:
                raise ValueError(f"unknown rule {rule}")

            rule = ProductNestedFixed([base_rule] * ndim)

    regions = []
    est = 0
    err = 0

    # Compute the estimates over the initial regions
    for a_k, b_k in initial_regions:
        # If any of the initial regions have zero width in one dimension, we can
        # ignore this as the integral will be 0 there.
        if not np.any(a_k == b_k):
            est_k = rule.estimate(f, a_k, b_k, args, kwargs)

            try:
                err_k = rule.estimate_error(f, a_k, b_k, args, kwargs)
            except NotImplementedError:
                raise ValueError("attempting cubature with a rule that doesn't \
implement error estimation.")

            regions.append(CubatureRegion(est_k, err_k, a_k, b_k))
            est += est_k
            err += err_k

    subdivisions = 0
    success = True

    with MapWrapper(workers) as mapwrapper:
        while np.any(err > atol + rtol * np.abs(est)):
            # region_k is the region with highest estimated error
            region_k = heapq.heappop(regions)

            est_k = region_k.estimate
            err_k = region_k.error

            a_k, b_k = region_k.a, region_k.b

            # Subtract the estimate of the integral and its error over this region from
            # the current global estimates, since these will be refined in the loop over
            # all subregions.
            est -= est_k
            err -= err_k

            # Find all 2^ndim subregions formed by splitting region_k along each axis,
            # e.g. for 1D integrals this splits an estimate over an interval into an
            # estimate over two subintervals, for 3D integrals this splits an estimate
            # over a cube into 8 subcubes.
            #
            # For each of the new subregions, calculate an estimate for the integral and
            # the error there, and push these regions onto the heap for potential
            # further subdividing.

            executor_args = zip(
                itertools.repeat(f),
                itertools.repeat(rule),
                itertools.repeat(args),
                itertools.repeat(kwargs),
                _subregion_coordinates(a_k, b_k),
            )

            for subdivision_result in mapwrapper(_process_subregion, executor_args):
                a_k_sub, b_k_sub, est_sub, err_sub = subdivision_result

                est += est_sub
                err += err_sub

                new_region = CubatureRegion(est_sub, err_sub, a_k_sub, b_k_sub)

                heapq.heappush(regions, new_region)

            subdivisions += 1

            if subdivisions >= max_subdivisions:
                success = False
                break

        status = "converged" if success else "not_converged"

        return CubatureResult(
            # Remember to multiply by sign determined from whether limits were correct
            # way around:
            estimate=sign*est,

            error=err,
            success=success,
            status=status,
            subdivisions=subdivisions,
            regions=regions,
            atol=atol,
            rtol=rtol,
        )


def _process_subregion(data):
    f, rule, args, kwargs, coord = data
    a_k_sub, b_k_sub = coord

    est_sub = rule.estimate(f, a_k_sub, b_k_sub, args, kwargs)
    err_sub = rule.estimate_error(f, a_k_sub, b_k_sub, args, kwargs)

    return a_k_sub, b_k_sub, est_sub, err_sub


def _subregion_coordinates(a, b, split_at=None):
    """
    Given the coordinates of a region like a=[0, 0] and b=[1, 1], yield the coordinates
    of all subregions split at a specific point (by default the midpoint), which in this
    case would be::

        ([0, 0], [1/2, 1/2]),
        ([0, 1/2], [1/2, 1]),
        ([1/2, 0], [1, 1/2]),
        ([1/2, 1/2], [1, 1])
    """

    if split_at is None:
        split_at = (a + b)/2

    for a_sub, b_sub in zip(
        itertools.product(*np.array([a, split_at]).T),
        itertools.product(*np.array([split_at, b]).T)
    ):
        yield np.array(a_sub), np.array(b_sub)


class VariableTransform:
    @property
    def limits(self):
        raise NotImplementedError

    @property
    def points(self):
        return []

    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError


class InfiniteLimitsTransform(VariableTransform):
    def __init__(self, f, a, b):
        self._f = f
        self._orig_a = a
        self._orig_b = b

        self._negate_pos = []
        self._semi_inf_pos = []
        self._double_inf_pos = []

        for i in range(len(a)):
            if a[i] == -np.inf and b[i] == np.inf:
                # (-oo, oo) will be mapped to (-1, 1)
                self._double_inf_pos.append(i)
            elif a[i] != -np.inf and b[i] == np.inf:
                # (start, oo) will be mapped to (0, 1)
                self._semi_inf_pos.append(i)
            elif a[i] == -np.inf and b[i] != np.inf:
                # (-oo, end) will be mapped to (0, 1)
                # This is handled by making the transformation t = -x and reducing it to
                # the other semi-infinite case.
                self._negate_pos.append(i)
                self._semi_inf_pos.append(i)

                # Since we flip the limits, we don't need to separately multiply the
                # integrand by -1.
                self._orig_a[i] = -b[i]
                self._orig_b[i] = -a[i]

        self._semi_inf_pos = np.array(self._semi_inf_pos)
        self._double_inf_pos = np.array(self._double_inf_pos)
        self._negate_pos = np.array(self._negate_pos)

    @property
    def limits(self):
        a, b = np.copy(self._orig_a), np.copy(self._orig_b)

        for index in self._double_inf_pos:
            a[index] = -1
            b[index] = 1

        for index in self._semi_inf_pos:
            a[index] = 0
            b[index] = 1

        return a, b

    @property
    def points(self):
        # If there are infinite limits, then the origin will be a problematic point
        # due to a division by zero there
        if self._double_inf_pos.size != 0 or self._semi_inf_pos.size != 0:
            return [np.zeros(self._orig_a.shape)]
        else:
            return []

    def __call__(self, t, *args, **kwargs):
        x = np.copy(t)

        if len(self._negate_pos) != 0:
            x[..., self._negate_pos] *= -1

        if self._double_inf_pos.size != 0:
            # For (-oo, oo) -> (-1, 1), use the transformation x = (1-|t|)/t.
            x[..., self._double_inf_pos] = (
                1 - np.abs(t[..., self._double_inf_pos])
            ) / t[..., self._double_inf_pos]

        if self._semi_inf_pos.size != 0:
            # For (start, oo) -> (0, 1), use the transformation x = start + (1 - t)/t.
            #
            # Need to expand start so it is broadcastable, and transpose to flip the
            # axis order.
            start = self._orig_a[self._semi_inf_pos][:, np.newaxis].T

            x[..., self._semi_inf_pos] = start + (
                1 - t[..., self._semi_inf_pos]
            ) / t[..., self._semi_inf_pos]

        f_x = self._f(x, *args, **kwargs)

        if self._double_inf_pos.size != 0:
            scale_factors = np.prod(
                t[..., self._double_inf_pos] ** 2,
                axis=-1,
            )
            scale_factors = scale_factors.reshape(-1, *([1]*(len(f_x.shape)-1)))

            f_x /= scale_factors

        if self._semi_inf_pos.size != 0:
            scale_factors = np.prod(
                t[..., self._semi_inf_pos] ** 2,
                axis=-1,
            )
            scale_factors = scale_factors.reshape(-1, *([1]*(len(f_x.shape)-1)))

            f_x /= scale_factors

        return f_x


class FuncLimitsTransform(VariableTransform):
    r"""
    Transform an integral with functions as limits to an integral with constant limits.

    Given an integral of the form:

    ..math ::

        \int^{b_1}_{a_1}
        \cdots
        \int^{b_n}_{a_n}
        \int^{B_1(x_1, \ldots, x_n)}_{A_0(x_1, \ldots, x_n)}
        \cdots
        \int^{B_m(x_1, \ldots, x_{n+m})}_{A_m(x_1, \ldots, x_{n+m})}
        f(x_1, \ldots, x_{n+m})
        dx_{n+m} \cdots dx_1

    an integral with :math:`n` outer non-function limits, and :math:`m` inner function
    limits, this will transform it into an integral over

        \int^{b_1}_{a_1}
        \cdots
        \int^{b_n}_{a_n}
        \int^{1}_{-1}
        \cdots
        \int^{1}_{-1}
        g(x_1, \ldots, x_n, y_1, \cdots, y_m)
        dy_m \cdots dy_1 dx_n \cdots dx_1

    Which is an integral over the original outer non-function limits and where a
    transformation has been applied so that the original function limits become [-1, 1].
    """

    def __init__(self, f, a, b, region):
        self._f = f
        self._a_outer = a
        self._b_outer = b

        self._region = region

        self._outer_ndim = len(self._a_outer)

        # Without evaluating the region at least once, it's impossible to know the
        # number of inner variables, which is required to return new limits.
        a_inner, _ = region(self._a_outer.reshape(1, -1))
        self._inner_ndim = np.array(a_inner).shape[-1]

    @property
    def limits(self):
        return (
            np.concatenate([self._a_outer, -np.ones(self._inner_ndim)]),
            np.concatenate([self._b_outer,  np.ones(self._inner_ndim)]),
        )

    def __call__(self, y, *args, **kwargs):
        a_inner, b_inner = self._region(y[:, :self._outer_ndim])

        # Allow specifying a_inner and b_inner as array_like rather than as ndarrays
        # since this is also allowed for a and b.
        a_inner = np.array(a_inner)
        b_inner = np.array(b_inner)

        npoints = y.shape[0]

        x = np.concatenate(
            [
                y[:, :self._outer_ndim],
                np.zeros((npoints, self._inner_ndim)),
            ],
            axis=-1,
        )

        for i in range(self._inner_ndim):
            a_i = a_inner[:, i]
            b_i = b_inner[:, i]

            x[:, self._outer_ndim + i] = (
                (b_i + a_i)/2 + (b_i - a_i)/2 * y[:, self._outer_ndim + i]
            )

        f_x = self._f(x, *args, **kwargs)
        jacobian = np.prod(b_inner - a_inner, axis=0) / 2**self._inner_ndim
        jacobian = jacobian.reshape(-1, *([1]*(len(f_x.shape) - 1)))

        return f_x * jacobian


def _is_in_region(point, a, b):
    if (point == a).all() or (point == b).all():
        return False
    return (a <= point).all() and (point <= b).all()


def _split_at_points(a, b, points):
    points = np.sort(points)

    for i, point in enumerate(points):
        if _is_in_region(point, a, b):
            regions = []
            points_ = np.delete(points, i, axis=0)

            for a_k, b_k in _subregion_coordinates(a, b, point):
                splits = _split_at_points(a_k, b_k, points_)

                if splits is not None:
                    regions.extend(splits)

            return regions

    return [(a, b)]


def _max_norm(x):
    return np.max(np.abs(x))
