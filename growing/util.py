class slice1d:
    def __init__(self, dim):
        self.dim = dim
    def __getitem__(self, s):
        return [slice(None)] * self.dim + [s]

def copy_block(src, dst, src_slice, dst_slice, dim=0):
    prefix = [slice(None)] * dim
    suffix = [slice(None)] * (src.dim() - dim)
    dst[prefix + dst_slice + suffix] = src[prefix + src_slice + suffix]
