import nnp.nn.fortran.nn_f95 as f95_api

def num_threads():
    """
    Return the number of threads running in parallel sections

    export omp_num_threads=X to change

    Examples
    --------
    >>> import nnp
    >>> print('running on {} threads'.format(nnp.util.misc.num_threads()))
    """

    return getattr(f95_api,"f90wrap_num_threads")()
