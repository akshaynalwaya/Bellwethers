"""
This kernel provide common interface for all kernels operating on
vectorial data.

This interface includes the following key kernel manipulations (functions):
    -- Dot(x1, x2): $K(x1, x2)$
    -- Expand(x1, x2, alpha): $sum_r K(x1_i,x2_r) \times alpha2_r$
    -- Tensor(x1, y1, x2, y2): $K(x1_i,x2_j) \times (y1_i \times y1_j)$
    -- TensorExpand(x1, y1, x2, y2, alpha2):
            $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$
    -- Remember(x): Remember data x
    -- Forget(x): Remove remembered data x

To design a specific kernel, simply overload these methods. The generic
kernel itself should never be instantiated, although methods
Remember and Forget are implemented in this class. The Remember method
stores the inner product of an input vector.

"""
from kernel import CKernel
import numpy


class CVectorKernel(CKernel):
    def __init__(self, blocksize=128):
        CKernel.__init__(self, blocksize)
        self._name = 'Vector kernel'
        ## @var _cacheKernel
        # Cache that store the base part for the kernel matrix.
        # This cache facilates the incremental and decremental
        # computational of the kernel matrix.
        #
        self._cacheKernel = {}
        ## @var _typicalParam
        # Typical parameter for the kernel. Many kernels do not
        # have parameters, such linear kernel. In this case, set
        # zero as the typical parameter. This variable will be
        # usefull when optimizing the kernel matrix with respect
        # to the kernel matrix.
        #
        self._typicalParam = numpy.array([0])

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

        ## Compute the kernel between the data points in x1 and those in x2.

    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        raise NotImplementedError, \
            'CVectorKernel.Dot in abstract class is not implemented'

        ## Compute the kernel between the data points in x1 and those in x2,

    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None):
        raise NotImplementedError, \
            'CVectorKernel.Expand in abstract class is not implemented'

        ## Compute the kernel between the data points in x1 and those in x2,

    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2. It returns a matrix
    # with entry $(ij)$ equal to $K(x1_i,x2_j) \times (y1_i \times y1_j)$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Tensor(self, x1, y1, x2, y2, index1=None, index2=None):
        raise NotImplementedError, \
            'CVectorKernel.Tensor in abstract class is not implemented'

        ## Compute the kernel between the data points in x1 and those in x2,

    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2, and final multiply
    # the resulting matrix by alpha2. It returns a matrix with entry $(ij)$
    # equal to $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def TensorExpand(self, x1, y1, x2, y2, alpha2, index1=None, index2=None, \
                     output=None):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

    ## Remember x by computing the inner product of the data points contained
    # in x, storing them in the cache and indexing them by the id of
    # x. If x have already been remembered,
    # the old stored information is simply overwritten.
    # @param x [read] The data to be remembered.
    #
    def Remember(self, x):
        # default behavior
        assert x is not None, 'x is None'
        assert len(x.shape) == 2, 'x is not a matrix'
        self._cacheData[id(x)] = (x ** 2).sum(axis=1)

    ## Remove the remembered data x. If x is not given, then all remembered
    # data in the cache is removed. If x has never been remembered, then
    # this function does nothing and return False instead.
    # @param x [read] The data to be removed.
    #
    def Forget(self, x=None):
        # default behavior
        if x is not None:
            assert len(x.shape) == 2, 'Argument 1 is has wrong shape'
            if self._cacheData.has_key(id(x)) is False:
                return False
            else:
                del self._cacheData[id(x)]
        else:
            self._cacheData.clear()

        return True

    ## Method that operates on the base part x of a kernel.
    # The derived classes overload this method to generate new
    # kernels.
    # @param x Base part of the kernel.
    #
    def Kappa(self, x):
        # default behavior
        return x

    ## Gradient of the kernel with respect to the kernel
    # parameter evaluated in the base part x of a kernel.
    # The derived classes overload this method to generate new
    # gradients of the kernels.
    # @param x Base part of the kernel.
    #
    def KappaGrad(self, x):
        # default behavior
        return numpy.zeros(x.shape)

    ## Function that set the parameter of the kernel.
    # If the derived classes have parameters, overload this
    # method to set the parameters.
    # @param param Parameters to be set.
    #
    def SetParam(self, param):
        # default behavior
        pass

    ## Clear the base part of the kernel computed for data x.
    # If x is not given, then all remembered data in the cache is removed.
    # If x has never been remembered, then this function does nothing
    # and return False instead.
    # @param x [read] The data whose base part is to be removed from the cache.
    #
    def ClearCacheKernel(self, x=None):
        # default behavior
        if x is not None:
            assert len(x.shape) == 2, "Argument 1 has wrong shape"
            if self._cacheKernel.has_key(id(x)) is False:
                return False
            else:
                del self._cacheKernel[id(x)]
        else:
            self._cacheKernel.clear()

        return True

    ## Create the cache for the base part of the kernel computed for
    # data x, and index them by the id of x. If x have already been
    # remembered, the old stored information is simply overwritten.
    # Overload this method to store different base part for different
    # kernels.
    # @param x [read] The data whose base part is to be cached.
    #
    def CreateCacheKernel(self, x):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

    ## Dot product of x with itself with the cached base part of the kernel.
    # Overload this method to use the base part differently for different
    # kernel. If param is given, the kernel matrix is computed using
    # the given parameter and the current base part. Otherwise, the old
    # parameters are used.
    # @param x The data set.
    # @param param The new parameters.
    # @param output The output buffer.
    #
    def DotCacheKernel(self, x, param=None, output=None):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2. Overload this method to define the decrement of the base
    # part differently for different kernels. Note that this method
    # updates the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    #
    def DecCacheKernel(self, x1, x2):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2, and return the resulting kernel matrix. If param is given,
    # the kernel matrix is computed using the given parameter and the
    # current base part. Otherwise, the old parameters are used. Overload
    # this method to have different behavior for different kernel. Note
    # that this method does NOT change the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    # @param param The new parameters.
    #
    def DecDotCacheKernel(self, x1, x2, param=None, output=None):
        raise NotImplementedError, \
            'CVectorKernel.K in abstract class is not implemented'

    ## Gradient of the kernel matrix with respect to the kernel parameter.
    # If param is given, the kernel matrix is computed using the given
    # parameter and the  current base part. Otherwise, the old parameters
    # are used. Overload this method to have different behavior.
    # @param x The data set for the kernel matrix.
    # @param param The kernel parameters.
    #
    def GradDotCacheKernel(self, x, param=None, output=None):
        # default behavior
        assert len(x.shape) == 2, "Argument 1 has wrong shape"
        assert self._cacheKernel.has_key(id(x)) == True, \
            "Argument 1 has not been cached"

        if param is not None:
            self.SetParam(param)

        n = x.shape[0]
        output = numpy.zeros((n, n), numpy.float64)
        return output
