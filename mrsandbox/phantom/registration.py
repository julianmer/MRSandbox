####################################################################################################
#                                         registration.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: Linear registration of one Volume onto another in cited numpy and scipy: the affine     #
#          that carries a measured map's scanner world into the virtual brain's world, so a        #
#          quantitative stack acquired on a subject reads at the phantom's positions.              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import scipy.ndimage
import scipy.optimize

from mrsandbox.phantom.volume import Volume


#**************#
#   sampling   #
#**************#
def _sample(volume, world):
    """The values of `volume` at world points (3, N), linearly interpolated, zero outside."""
    index = np.linalg.solve(volume.affine[:3, :3], world - volume.affine[:3, 3:4])
    return scipy.ndimage.map_coordinates(volume.data, index, order=1, mode="constant", cval=0.0)


#****************#
#   parameters   #
#****************#
def _matrix(params, centre):
    """The 4x4 world transform of twelve parameters about `centre`: translation (mm),
    rotation (degrees about x, y, z), scale, and shear."""
    t, r, s, k = (np.asarray(params[i:i + 3], dtype=np.float64) for i in (0, 3, 6, 9))
    cx, cy, cz = np.cos(np.radians(r))
    sx, sy, sz = np.sin(np.radians(r))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    shear = np.array([[1, k[0], k[1]], [0, 1, k[2]], [0, 0, 1]])
    linear = rz @ ry @ rx @ shear @ np.diag(s)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = t + centre - linear @ centre
    return matrix


#**************#
#   register   #
#**************#
def register(source, target, coarse_mm=2.4, smooth_mm=2.0):
    """The world-to-world affine carrying `source` onto `target`, both Volumes of one
    quantity (or of monotonically related contrasts): the twelve-parameter linear map
    (translation, rotation, anisotropic scale, shear about the target's centroid)
    that maximizes the normalized cross-correlation between the target's values over
    its support and the source's values sampled at the mapped positions, zero where
    the source has none, so nothing rewards sliding the source off the target.
    Both are Gaussian smoothed by `smooth_mm` for the fit (a speckled quantitative
    map otherwise gives a flat correlation landscape). Initialized by the two
    centroids; fitted by Powell's method (scipy) on a grid subsampled to about
    `coarse_mm`, translation and scale first, then everything, a quadratic penalty
    keeping the map a plausible head-to-head transform (scale 0.7 to 1.4, shear
    within 0.3, rotation within 45 degrees, translation within 100 mm of the
    centroid match). Returns the 4x4 matrix T with world_target = T @ world_source, so
    Volume(source.data, T @ source.affine) is the source in the target's world.
    """
    def smoothed(volume):
        sigma = smooth_mm / np.linalg.norm(volume.affine[:3, :3], axis=0)
        return Volume(scipy.ndimage.gaussian_filter(volume.data.astype(np.float32), sigma),
                      volume.affine)

    source_fit, target_fit = smoothed(source), smoothed(target)
    step = np.maximum(np.rint(coarse_mm / np.linalg.norm(target.affine[:3, :3], axis=0)),
                      1).astype(int)
    grid = np.array(np.nonzero(target.data[::step[0], ::step[1], ::step[2]] > 0.0)) * step[:, None]
    world = target.affine[:3, :3] @ grid + target.affine[:3, 3:4]
    values = target_fit.data[tuple(grid)].astype(np.float64)
    values = values - values.mean()
    centre = world.mean(axis=1)
    support = np.array(np.nonzero(source.data > 0.0)).astype(np.float64)
    source_centre = source.affine[:3, :3] @ support.mean(axis=1) + source.affine[:3, 3]

    shift = centre - source_centre
    base = np.array([*shift, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # the plausible ranges, as a quadratic penalty beyond them (a hard bound trips
    # Powell's line searches): translation 100 mm about the centroid match, rotation
    # 45 degrees, scale 0.7 to 1.4, shear 0.3
    low = np.array([*(shift - 100.0), -45.0, -45.0, -45.0, 0.7, 0.7, 0.7, -0.3, -0.3, -0.3])
    high = np.array([*(shift + 100.0), 45.0, 45.0, 45.0, 1.4, 1.4, 1.4, 0.3, 0.3, 0.3])

    def cost(params, free, base):
        full = np.array(base)
        full[free] = params
        matrix = _matrix(full, centre)
        sampled = _sample(source_fit, np.linalg.solve(
            matrix, np.vstack([world, np.ones((1, world.shape[1]))]))[:3])
        sampled = sampled - sampled.mean()
        norm = np.linalg.norm(sampled) * np.linalg.norm(values)
        outside = np.maximum(low - full, 0.0) + np.maximum(full - high, 0.0)
        penalty = np.sum((outside / (high - low)) ** 2) * 100.0
        return (1.0 - (sampled @ values) / norm if norm > 0.0 else 1.0) + penalty

    for free in ([0, 1, 2, 6, 7, 8], list(range(12))):
        result = scipy.optimize.minimize(cost, base[free], args=(free, base), method="Powell",
                                         options={"xtol": 1e-3, "ftol": 1e-6})
        base[free] = result.x
    return _matrix(base, centre)


#**************#
#   overlap    #
#**************#
def overlap(source, target, matrix):
    """The Dice overlap of the two supports (values above zero) with `matrix` applied
    to the source, evaluated on the target's grid: a registration's quality figure."""
    grid = np.array(np.nonzero(np.ones(target.shape[:3], dtype=bool)))
    world = target.affine[:3, :3] @ grid + target.affine[:3, 3:4]
    mapped = np.linalg.solve(matrix, np.vstack([world, np.ones((1, world.shape[1]))]))[:3]
    inside = _sample(Volume((source.data > 0.0).astype(np.float32), source.affine), mapped) > 0.5
    target_inside = target.data[tuple(grid)] > 0.0
    return 2.0 * np.sum(inside & target_inside) / (inside.sum() + target_inside.sum())
