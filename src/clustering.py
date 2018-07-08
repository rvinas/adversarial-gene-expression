class Cluster:
    """
    Auxiliary class to store binary clusters
    """

    def __init__(self, c_left=None, c_right=None, index=None):
        assert (index is not None) ^ (c_left is not None and c_right is not None)
        self._c_left = c_left
        self._c_right = c_right
        if index is not None:
            self._indices = [index]
        else:
            self._indices = c_left.indices + c_right.indices

    @property
    def indices(self):
        return self._indices

    @property
    def c_left(self):
        return self._c_left

    @property
    def c_right(self):
        return self._c_right


if __name__ == '__main__':
    pass
