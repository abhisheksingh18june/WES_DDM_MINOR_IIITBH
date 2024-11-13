import torch.nn as nn

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.

    Args:
        dims (int): The number of dimensions of the convolution.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The convolution module.

    Raises:
        ValueError: If `dims` is not one of 1, 2, or 3.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# Difference between args and kwargs
'''
args is a tuple of positional arguments, while kwargs is a dictionary of keyword arguments.
#example
args = (1, 2, 3)
kwargs = {'a': 1, 'b': 2}
print(args)  # Output: (1, 2, 3)
print(kwargs)  # Output: {'a': 1, 'b': 2}

Position arguments are passed to a function in the order in which the parameters are defined in the function's signature.

Keyword arguments are passed to a function using the parameter name as the key, followed by the value.they don't need to be in the same order as the parameters are defined in the function's signature.
'''