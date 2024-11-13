import torch.nn as nn

def avg_pool_nd(
    dims: int,
    *args,
    **kwargs,
) -> nn.Module:
    """
    Create a 1D, 2D, or 3D average pooling module.

    Args:
        dims: The number of dimensions of the average pooling module.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The average pooling module.

    Raises:
        ValueError: If `dims` is not one of 1, 2, or 3.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError("unsupported dimensions: {}".format(dims))

def max_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D max pooling module.

    Args:
        dims: The number of dimensions of the max pooling module.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The max pooling module.

    Raises:
        ValueError: If `dims` is not one of 1, 2, or 3.
    """
    # TODO: Allow specifying the `padding` argument for 3D max pooling
    if dims == 1:
        return nn.MaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    elif dims == 3:
        if 'padding' in kwargs:
            padding = kwargs['padding']
            del kwargs['padding']
        else:
            padding = 0
        return nn.MaxPool3d(*args, padding=padding, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

