import numpy as np
import torch as th
import torch.nn as nn

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

if __name__ == "__main__":
    import sys; sys.exit(0)
    conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
    ''' 
    in_channels: number of input channels like 3 in case of rgb images
    out_channels: number of output channels (Number of Filters)
    kernel_size: size of the convolutional kernel
     
    Equivalent TensorFlow code:
    conv_layer = tf.keras.layers.Conv2D(
    filters=1,             # Equivalent to out_channels in PyTorch
    kernel_size=3,         # Same kernel size
    strides=1,             # Default stride is 1 (same as PyTorch's default)
    padding='valid',       # 'valid' padding is the default, which is the same as PyTorch's default
    input_shape=(None, None, 3)  # Specify 3 input channels (for RGB input)
    )
    '''
    print("Original data type:", conv_layer.weight.data.dtype)
    print(conv_layer.weight.data)
    
    # Convert to float16
    convert_module_to_f16(conv_layer)
    print("Data type after convert_module_to_f16:", conv_layer.weight.data.dtype)
    
    # Convert back to float32
    convert_module_to_f32(conv_layer)
    print("Data type after convert_module_to_f32:", conv_layer.weight.data.dtype)