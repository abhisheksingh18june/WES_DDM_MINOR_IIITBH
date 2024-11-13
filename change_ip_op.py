import torch.nn as nn

def change_input_output_unet(model, in_channels=4, out_channels=8):
    """

    :param model: unet model from guided diffusion code, for 256x256 image input
    :param in_channels:
    :param out_channels:
    :return: the model with the change
    """

    # change the input
    kernel_size = model.input_blocks[0][0].kernel_size
    stride = model.input_blocks[0][0].stride
    padding = model.input_blocks[0][0].padding
    out_channels_in = model.input_blocks[0][0].out_channels
    model.input_blocks[0][0] = nn.Conv2d(in_channels, out_channels_in, kernel_size, stride, padding)

    # change the input
    kernel_size = model.out[-1].kernel_size
    stride = model.out[-1].stride
    padding = model.out[-1].padding
    in_channels_out = model.out[-1].in_channels
    model.out[-1] = nn.Conv2d(in_channels_out, out_channels, kernel_size, stride, padding)

    return model