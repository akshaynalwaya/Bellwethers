"""
This module perform computation related to Hilber-Schmidt Independence
Criterion. Hilber-Schmidt Independence Criterion is short for HSIC.

HISC is defined as $HSIC=\frac{1}{m}Tr(KHLH)$, where $kMat$ and $lMat$
are the kernel matrices for the data and the labels respectively.
$H=I-\frac{1}{m}\delta_{ij}$, where $m$ is the number of data points,
is the centering matrix. The unbiased estimator of HSIC is computed as
$HSIC=\frac{1}{m(m-3)}\left[Tr(KL)+\frac{1}{(m-1)(m-2)}1^\top K11^\top L1
-\frac{2}{m-2}1^\top KL1\right]. For more theoretical underpinning
of HSIC, see the following reference:

Gretton, A., O. Bousquet, A. Smola and B. Schoelkopf: Measuring
Statistical Dependence with Hilbert-Schmidt Norms. Algorithmic
Learning Theory: 16th International Conference, ALT 2005, 63-78, 2005.
"""

import numpy
import vector
from utils import setdiag0


class CHSIC(object):
    def __init__(self):
        pass

    def ComputeHLH(self, y, kernely=vector.CLinearKernel()):
        """
        Compute HLH give the labels.
            @param y The labels.
            @param kernely The kernel on the labels, default to linear kernel.
        """
        ny = y.shape
        if len(ny) > 1:
            lMat = kernely.Dot(y, y)
        else:
            lMat = numpy.outerproduct(y, y)

        sL = numpy.sum(lMat, axis=1)
        ssL = numpy.sum(sL)
        # hlhMat
        return lMat - numpy.add.outer(sL, sL) / ny[0] + ssL / (ny[0] * ny[0])

    def BiasedHSIC(self, x, y, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel()):
        """
        Compute the biased estimator of HSIC.
            @param x The data.
            @param y The labels.
            @param kernelx The kernel on the data, default to linear kernel.
            @param kernely The kernel on the labels, default to linear kernel.
        """

        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
            "Argument 1 and 2 have different number of data points"

        if len(nx) > 1:
            kMat = kernelx.Dot(x, x)
        else:
            kMat = numpy.outerproduct(x, x)

        hlhMat = self.ComputeHLH(y, kernely)
        return numpy.sum(numpy.sum(kMat * hlhMat)) / ((nx[0] - 1) * (nx[0] - 1))

    def BiasedHSICFast3(self, x, y):
        """
        Fast computation of the biased HSIC when the kernel matrix
        for the data K can be decomposed into K = x * x' and that
        for the labels can be decomposed into HLH = y * y' and the
        rank of y is low (this will be useful after incomplete cholesky
        factorization.
        Args:
            x: The K = x * x' for the data.
            y: The HLH = y * y' for the labels.

        Returns:
            HSIC score
        """
        nx = x.shape
        assert (x.shape[0] == y.shape[0]), \
            "Argument 1 and 2 have different shapes"

        return (numpy.dot(x.T, y) ** 2).sum() / ((nx[0] - 1) * (nx[0] - 1))

    def UnBiasedHSIC(self, x, y, kernelx=vector.CLinearKernel(), \
                     kernely=vector.CLinearKernel()):
        """
        Compute the UNbiased estimator of HSIC.

        Args:
            x: The data.
            y: The labels.
            kernelx: The kernel on the data, default to linear kernel.
            kernely: The kernel on the labels, default to linear kernel.

        Returns:
            HSIC score
        """
        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
            "Argument 1 and 2 have different number of data points"

        kMat = kernelx.Dot(x, x)
        setdiag0(kMat)

        lMat = kernely.Dot(y, y)
        setdiag0(lMat)

        sK = kMat.sum(axis=1)
        ssK = sK.sum()
        sL = lMat.sum(axis=1)
        ssL = sL.sum()

        return (kMat.__imul__(lMat).sum() + \
                (ssK * ssL) / ((nx[0] - 1) * (nx[0] - 2)) - \
                2 * sK.__imul__(sL).sum() / (nx[0] - 2) \
                ) / (nx[0] * (nx[0] - 3))

    def UnBiasedHSICFast(self, kMat, lMat, sL, ssL):
        """
        Fast computation of the biased HSIC when the kernel matrix
        for the data and the HLH matrix for the labels are already
        computed.

        Args:
            kMat: The kernel matrix for the data.
            lMat: The kernel matrix of the label.
            sL: The vector of the sum of each row of lMat.
            ssL: The vector of the sum of all entries in lMat.

        Returns:
            HSIC score
        """
        nx = kMat.shape
        assert (kMat.shape == lMat.shape), \
            "Argument 1 and 2 have different shapes"

        sK = numpy.sum(kMat, axis=1)
        ssK = numpy.sum(sK)

        return (numpy.sum(numpy.sum(kMat * lMat)) \
                + (ssK * ssL) / ((nx[0] - 1) * (nx[0] - 2)) \
                - 2 * numpy.sum(sK * sL) / (nx[0] - 2) \
                ) / (nx[0] * (nx[0] - 3))


def normalize(data):
    """
    Normalize each dimension of the data separately to zero mean and unit
    standard deviation.

    Args:
        data: The data to be normalized. Each row is a datum and each column a dimension.

    Returns:
        None
    """
    m = data.mean(axis=0)
    s = data.std(axis=0)
    data.__isub__(m).__itruediv__(s)


def center(k):
    """
    Center the kernel matrix in the feature space.

    Args:
        k: The kernel matrix to be centered.

    Returns:
        None
    """
    n = k.shape
    assert n[0] == n[1], 'k must be symmetric and positive semidefinite'
    mk = k.mean(axis=1)
    mk.shape = (n[0], 1)
    mmk = mk.mean()
    k.__isub__(mk).__isub__(mk.T).__iadd__(mmk)
