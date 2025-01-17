
import sys
import os
from os.path import join as pjoin
import numpy as np
import yaml
import argparse
import datetime
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torchvision.transforms.functional as tvtf


# %% image functions

def min_max_norm(img, global_norm=True, is_uint8=True):
    """
    assume input is a torch tensor [3,h,w]
    """
    if global_norm:
        img_norm = img - img.min()
        img_norm /= img_norm.max()
    else:
        img_norm = torch.zeros_like(img)

        img_norm[0, :, :] = img[0, :, :] - img[0, :, :].min()
        img_norm[1, :, :] = img[1, :, :] - img[1, :, :].min()
        img_norm[2, :, :] = img[2, :, :] - img[2, :, :].min()

        img_norm[0, :, :] = img_norm[0, :, :] / img_norm[0, :, :].max()
        img_norm[1, :, :] = img_norm[1, :, :] / img_norm[1, :, :].max()
        img_norm[2, :, :] = img_norm[2, :, :] / img_norm[2, :, :].max()

    if is_uint8:
        img_norm *= 255
        img_norm = img_norm.to(torch.uint8)

    return img_norm


def min_max_norm_range(img, vmin=0, vmax=1, is_uint8=False):
    """
    assume input is a torch tensor [3/1,h,w] or [Batch,3/1,h,w]
    """

    vmin = float(vmin)
    vmax = float(vmax)

    # Compute the minimum and maximum values for each image in the batch separately
    if len(img.shape) == 4:
        # support a batch
        img_min = img.view(img.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        img_max = img.view(img.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)

    elif len(img.shape) == 3:
        img_max = img.max()
        img_min = img.min()

    else:
        raise NotImplementedError

    if img_min == img_max:
        img_norm = torch.zeros_like(img)
    else:
        scale = (vmax - vmin) / (img_max - img_min)
        img_norm = (img - img_min) * scale + vmin

    if is_uint8:
        img_norm = (255 * img_norm).to(torch.uint8)

    return img_norm


def min_max_norm_range_percentile(img, vmin=0, vmax=1, percent_low=0., percent_high=1., is_uint8=False):
    """
    assume input is a torch tensor [3/1,h,w]
    """

    # first clip into percentile values
    img_min = torch.quantile(img, q=percent_low)
    img_max = torch.quantile(img, q=percent_high)
    img_clip = torch.clamp(img, img_min, img_max)

    vmin = float(vmin)
    vmax = float(vmax)

    # Compute the minimum and maximum values for each image in the batch separately
    if len(img_clip.shape) == 4:
        # support a batch
        img_min = img_clip.view(img.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        img_max = img_clip.view(img.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)

    elif len(img.shape) == 3:
        img_max = img_clip.max()
        img_min = img_clip.min()

    else:
        raise NotImplementedError

    if img_min == img_max:
        img_norm = torch.zeros_like(img_clip)
    else:
        scale = (vmax - vmin) / (img_max - img_min)
        img_norm = (img_clip - img_min) * scale + vmin

    if is_uint8:
        img_norm = (255 * img_norm).to(torch.uint8)

    return img_norm


def max_norm(img, global_norm=True, is_uint8=True):
    """
    assume input is a torch tensor [3,h,w]
    """

    if global_norm:
        img_norm = img / img.max()

    else:
        img_norm = torch.zeros_like(img)
        img_norm[0, :, :] = img[0, :, :] / img[0, :, :].max()
        img_norm[1, :, :] = img[1, :, :] / img[1, :, :].max()
        img_norm[2, :, :] = img[2, :, :] / img[2, :, :].max()

    if is_uint8:
        img_norm *= 255
        img_norm = img_norm.to(torch.uint8)

    return img_norm


def clip_image(img, scale=True, move=True, is_uint8=True):
    """
    assume input is a torch tensor [ch,h,w]
    ch can be 3/1
    """

    # fix in case the image is only [imagesize, imagesize]
    if len(img.shape) == 2:
        img = img.unsqueeze(0)

    if move:
        img = img + 1
    if scale:
        img = 0.5 * img

    if is_uint8:
        img *= 255
        img = img.clamp(0, 255).to(torch.uint8)
    else:
        img = img.clamp(0, 1)

    return img


def gaussian_kernel(kernel_size, sigma=1., muu=0.):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(0, kernel_size, kernel_size),
                       np.linspace(0, kernel_size, kernel_size))

    x -= kernel_size // 2
    y -= kernel_size // 2

    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    # normal = 1 / (2 * np.pi * sigma ** 2)
    normal = 1

    # Calculating Gaussian filter
    # gauss = normal * np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
    gauss = normal * np.exp(-((dst - muu) ** 2 / (sigma ** 2)))

    return gauss


def create_image_text_to_grid(image, image_size=[256, 256], info_str="light factor", norm=True):
    """
    input is an image (1 or 3 channels) or a scalar (1 or 3 channels) as pytorch tensor
    outputs are:
    1. a tensor image [3, image_size, image_size]
    2. a text for logger
    """
    shape = list(image.detach().cpu().shape)
    image = image.detach().cpu()

    # 1 channel scalar
    if len(image.detach().cpu().shape) == 1:
        text = f"{info_str} = {image.item():.3f}"
        out_image = image * torch.ones(size=[3] + image_size)
        # cast to uint8
        out_image = (255 * out_image).to(torch.uint8)

    # 3 channels scalar
    elif len(shape) == 3 and shape[-3] == 3 and (not shape[-2::] == image_size):
        text = f"{info_str} = " \
               f"[{image.squeeze().numpy()[0]:.3f}, " \
               f"{image.squeeze().numpy()[1]:.3f}, " \
               f"{image.squeeze().numpy()[2]:.3f}]"
        out_image = torch.zeros(size=[3, image_size[0], image_size[1]], dtype=torch.float32)
        out_image[0] = image[0] * torch.ones(size=image_size)
        out_image[1] = image[1] * torch.ones(size=image_size)
        out_image[2] = image[2] * torch.ones(size=image_size)
        # cast to uint8
        out_image = (255 * out_image).to(torch.uint8)

    # 1 channel image
    elif len(shape) == 2 and shape[-2::] == image_size:
        text = f"{info_str} mean = {image.mean():.3f}\n" \
               f"{info_str} std = {image.std():.3f}\n" \
               f"{info_str} min = {image.min():.3f}\n" \
               f"{info_str} max = {image.max():.3f}"
        out_image = image.unsqueeze(0).repeat(3, 1, 1)
        out_image = min_max_norm(out_image, is_uint8=True) if norm else (255 * out_image).to(torch.uint8)

    # 3 channels image

    elif len(shape) == 3 and shape[-3] == 3 and shape[-2::] == image_size:
        text = f"Red mean={image[0].mean():.3f}, std={image[0].std():.3f}, min={image[0].min():.3f}, max={image[0].max():.3f}\n" \
               f"Green mean={image[1].mean():.3f}, std={image[1].std():.3f}, min={image[1].min():.3f}, max={image[1].max():.3f}\n" \
               f"Blue mean={image[2].mean():.3f}, std={image[2].std():.3f}, min={image[2].min():.3f}, max={image[2].max():.3f}\n"
        out_image = min_max_norm(image, is_uint8=True) if norm else (255 * image).to(torch.uint8)

    else:
        ValueError(f"Image dimensions are not recognized - shape={shape}")

    return out_image, text


def add_text_torch_img(img, text, font_size=15):
    """

    :param img: torch image shape [3,h,w]
    :param text: text to insert
    :return: torch image shape [3,h,w]
    """

    # print betas and b_inf on b_inf image
    img_pil = tvtf.to_pil_image(img)
    I_text = ImageDraw.Draw(img_pil)

    if sys.platform.startswith("win"):
        I_text.font = ImageFont.truetype("arial.ttf", font_size)
    elif sys.platform.startswith("linux"):
        I_text.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size,
                                         encoding="unic")
    else:
        raise NotImplementedError
    I_text.multiline_text((5, 30), text, fill=(0, 0, 0))
    b_inf_image = tvtf.to_tensor(img_pil)

    return b_inf_image


# %% change input and outputs of the unet

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


# %% masked mse loss

class MaskedMSELoss(_Loss):
    """

    masked version of MSE loss

    """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MaskedMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_base_loss = mask * F.mse_loss(input, target, reduction='none')

        if self.reduction == 'sum':
            masked_mse_loss = masked_base_loss.sum()
        elif self.reduction == 'mean':
            # the number of channel of mask is 1, and the image the RGB/RGBD therefore a multiplication is required
            num_channels = input.shape[1]
            num_non_zero_elements = num_channels * mask.sum()
            masked_mse_loss = masked_base_loss.sum() / num_non_zero_elements
        elif self.reduction == 'none':
            masked_mse_loss = masked_base_loss
        else:
            ValueError(f"Unknown reduction input: {self.reduction}")

        return masked_mse_loss


# %% masked L1 loss

class MaskedL1Loss(_Loss):
    """

    masked version of L1 loss

    """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MaskedL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_base_loss = mask * F.l1_loss(input, target, reduction='none')

        if self.reduction == 'sum':
            masked_l1_loss = masked_base_loss.sum()
        elif self.reduction == 'mean':
            # the number of channel of mask is 1, and the image the RGB/RGBD therefore a multiplication is required
            num_channels = input.shape[1]
            num_non_zero_elements = num_channels * mask.sum()
            masked_l1_loss = masked_base_loss.sum() / num_non_zero_elements
        elif self.reduction == 'none':
            masked_l1_loss = masked_base_loss
        else:
            ValueError(f"Unknown reduction input: {self.reduction}")

        return masked_l1_loss


# %% read yaml config file and parser functions

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# read yaml file (config file and write the content into txt file)

def yaml_to_txt(yaml_file_path, txt_file_path):
    # Read YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Convert YAML data to a string
    yaml_text = yaml.dump(yaml_data, default_flow_style=False)

    # Write YAML data to a text file
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(yaml_text)


# dictionary and argparser functions

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    function from guided diffusion code
    """

    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def add_dict_to_namespace(namespace, args_dict):
    for key, value in args_dict.items():
        setattr(namespace, key, value)


# save directory using date
def update_save_dir_date(arguments_save_dir: str) -> str:
    today = datetime.date.today()
    today = f"{today.day}-{today.month}-{today.year % 2000}"
    run_description = "run1"
    save_dir = pjoin(arguments_save_dir, f"{today}", run_description)

    # check if this path is already exist
    while True:
        if os.path.exists(save_dir):

            digits = re.findall(r'\d+$', save_dir)[0]
            digits_len = len(str(digits))
            save_dir = f"{save_dir[0:-digits_len]}{int(digits) + 1}"
        else:
            break
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


# checkpoint path update

def update_checkpoint_path(save_dir_path: str) -> str:
    checkpoint_path = os.path.join(save_dir_path, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)
    return os.path.join(checkpoint_path, "checkpoint.pt")


# update_relevant_arguments function

def update_relevant_arguments(args, save_dir_path: str):
    # cast to float relevant inputs
    args.lr, args.fp16_scale_growth = float(args.lr), float(args.fp16_scale_growth)
    args.save_dir = update_save_dir_date(args.save_dir_main)
    args.checkpoint_path = update_checkpoint_path(args.save_dir) if args.save_checkpoint else ""

    # specify number of input and output channels according to pretrained model
    if args.pretrain_model == "debka":
        args.unet_in_channels = 4
        args.unet_out_channels = (4 if not args.learn_sigma else 8)
    else:
        args.unet_in_channels = 3
        args.unet_out_channels = (3 if not args.learn_sigma else 6)

    return args


# arguments parser functions
def arguments_from_file(config_file_path: str) -> argparse.Namespace:
    # read config file
    args_dict = load_yaml(config_file_path)

    # create argparse Namspace object
    args = argparse.Namespace()

    # add config dictionary into argparse namespace
    add_dict_to_namespace(args, args_dict)

    return args


# os run

def get_os():
    if sys.platform.startswith('linux'):
        os_run = 'linux'
    elif sys.platform.startswith('win'):
        os_run = 'win'
    else:
        print("Running on a different platform")

    return os_run


# %% return torch optimizer by name

def get_optimizer(optimizer_name, model_parameters, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name is None or optimizer_name == "gd" or optimizer_name == "":
        return None
    elif optimizer_name == 'adam':
        return optim.Adam(model_parameters, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, **kwargs)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model_parameters, **kwargs)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model_parameters, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, **kwargs)
    elif optimizer_name == 'sparseadam':
        return optim.SparseAdam(model_parameters, **kwargs)
    elif optimizer_name == 'adamax':
        return optim.Adamax(model_parameters, **kwargs)
    elif optimizer_name == 'asgd':
        return optim.ASGD(model_parameters, **kwargs)
    elif optimizer_name == 'lbfgs':
        return optim.LBFGS(model_parameters, **kwargs)
    elif optimizer_name == 'rprop':
        return optim.Rprop(model_parameters, **kwargs)
    elif optimizer_name == 'rprop':
        return optim.Rprop(model_parameters, **kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")


# %% change depth function according to the input of depth type

def get_depth_value(value_raw, **kwargs):
    if isinstance(value_raw, float):
        value = value_raw
    elif isinstance(value_raw, int):
        value = float(value_raw)
    elif isinstance(value_raw, str):
        value = np.fromstring(value_raw, dtype=float, sep=',')
    elif isinstance(value_raw, (np.ndarray, np.generic)):
        value = value_raw
    else:
        raise NotImplementedError

    return value


def convert_depth(depth, depth_type, **kwargs):
    """

    :param depth: expected to get the depth as it gets out from the unet model
    :param depth_type: the type of conversion
    :return: converted out depth
    """
    tmp_value = kwargs.get("value", None)
    value = get_depth_value(tmp_value)

    if depth_type == "move":
        depth_out = depth + value

    elif depth_type == "gamma":
        depth_out = torch.pow((depth + value[0]) * value[1], value[2])

    elif depth_type is None or depth_type == "original":
        depth_out = 0.5 * (depth + 1.0)

    else:
        raise NotImplementedError

    return depth_out


# %% when pattern sampling - check if freezing phi is required

def is_freeze_phi(sample_pattern, time_index, num_timesteps):
    # original sampling (no freezing phi required at all)
    if (sample_pattern is None) or (sample_pattern["pattern"] == "original"):
        freeze_phi = False

        # in case of non guidance for that time index, no alternating happens
    elif time_index > sample_pattern['start_guidance'] * num_timesteps or \
            time_index < sample_pattern['stop_guidance'] * num_timesteps:
        freeze_phi = True

    # gibbsDDRM pattern sampling - but before starting update phi
    elif time_index > sample_pattern["update_start"] * num_timesteps or time_index < sample_pattern[
        "update_end"] * num_timesteps:
        freeze_phi = True

    # otherwise not freezing phi
    else:
        freeze_phi = False

    return freeze_phi


# %% when pattern sampling - set alternating length

def set_alternate_length(sample_pattern, time_index, num_timesteps):
    # check correction of the values
    if (sample_pattern["pattern"] != "original") and (sample_pattern is not None):

        assert sample_pattern["update_start"] > sample_pattern["update_end"]
        assert sample_pattern["s_start"] > sample_pattern["s_end"]

        if sample_pattern['local_M'] > 1:
            assert sample_pattern["update_start"] >= sample_pattern["s_start"]
            assert sample_pattern["s_end"] >= sample_pattern["update_end"]

        # this is the original - non pattern case
    if (sample_pattern is None) or (sample_pattern["pattern"] == "original"):
        alternate_length = 1

    # in case of non guidance for that time index, no alternating happens
    elif time_index > sample_pattern['start_guidance'] * num_timesteps or \
            time_index < sample_pattern['stop_guidance'] * num_timesteps:
        alternate_length = 1

    # Until start update there is no optimization of phi's - This is mentioned in the gibbsDDRM paper
    elif time_index > sample_pattern["update_start"] * num_timesteps or \
            time_index < sample_pattern["update_end"] * num_timesteps:
        alternate_length = 1

    # PGDiff paper - S_start and S_end - time indices which the alternate optimization is happened
    # in this case S_start should be smaller than update start
    # s_start should be smaller than update_start and s_end larger than update_end
    elif time_index > sample_pattern["s_start"] * num_timesteps or \
            time_index < sample_pattern["s_end"] * num_timesteps:
        alternate_length = 1

    else:
        alternate_length = sample_pattern["local_M"]

    return alternate_length


# %% logging text

def log_text(args):
    log_txt_tmp = f"\n\nGuidance Scale: {args.conditioning['params']['scale']}" \
                  f"\nLoss Function: {args.conditioning['params']['loss_function']}" \
                  f"\nweight: {args.conditioning['params']['loss_weight']}, " \
                  f"weight_function: {args.conditioning['params']['weight_function']}" \
                  f"\nAuxiliary Loss: {args.aux_loss['aux_loss']}" \
                  f"\nUnderwater model: {args.measurement['operator']['name']}" \
                  f"\nOptimize w.r.t: {'x_prev' if args.conditioning['params']['gradient_x_prev'] else 'x0'}" \
                  f"\nOptimizer model: {args.measurement['operator']['optimizer'] if 'optimizer' in list(args.measurement['operator'].keys()) else 'none'}, " \
                  f"\nManual seed: {args.manual_seed}" \
                  f"\nDepth type: {args.measurement['operator']['depth_type']}, value: {args.measurement['operator']['value']}"

    log_noise_txt = f"\nNoise: {args.measurement['noise']['name']}"
    if 'sigma' in list(args.measurement['noise'].keys()):
        log_noise_txt += f", sigma: {args.measurement['noise']['sigma']}"
    log_txt_tmp += log_noise_txt

    gradient_clip_tmp = args.conditioning['params']['gradient_clip']
    gradient_clip_tmp = [num_str for num_str in gradient_clip_tmp.split(',')]
    log_grad_clip_txt = f"\nGradient Clipping: {gradient_clip_tmp[0]}"
    gradient_clip = str2bool(gradient_clip_tmp[0])
    if gradient_clip:
        log_grad_clip_txt += f", min value: -{gradient_clip_tmp[1]}, max value: {gradient_clip_tmp[1]}"
    log_txt_tmp += log_grad_clip_txt

    if args.sample_pattern['pattern'] == 'original':
        log_txt_tmp += f"\nSample Pattern: original"
    else:
        log_txt_tmp += f"\nSample Pattern: {args.sample_pattern['pattern']}, " \
                       f"\n     Guidance start: {args.sample_pattern['start_guidance']} ,end: {args.sample_pattern['stop_guidance']}" \
                       f"\n     Optimizations iters: {args.sample_pattern['n_iter']}, " \
                       f"\n     Update start from: {args.sample_pattern['update_start']}, end: {args.sample_pattern['update_end']}" \
                       f"\n     M: {args.sample_pattern['local_M']}, start: {args.sample_pattern['s_start']}, end: {args.sample_pattern['s_end']}"

    return log_txt_tmp


# %% loss_weight - factor the difference between the  measurement to the degraded image

def set_loss_weight(loss_weight_type, weight_function=None, degraded_image=None, x_0_hat=None):
    # weight function is a string divided into "function,value0,value1,..."
    if isinstance(weight_function, str):
        str_parts = weight_function.split(",")
        function_str = str_parts[0]

        if len(str_parts) > 1:
            value = np.asarray(str_parts[1:]).astype(float)
            value = value.item() if value.shape[0] == 1 else value

    else:
        function_str = 'none'

    if loss_weight_type == 'none' or loss_weight_type is None:
        loss_weight = 1

    # try to multiply by the depth, the reason is to make the gradients of the far area larger since the
    # prediction from the u-net got close to zero at those areas
    elif loss_weight_type == 'depth':

        depth_tmp = x_0_hat.detach()[:, 3, :, :].unsqueeze(1)
        loss_weight = convert_depth(depth=depth_tmp, depth_type=function_str, value=value)

    else:
        raise NotImplementedError

    return loss_weight


# %% create histogram image

def color_histogram(img, title=None):
    """
    :param img: image should be tensor (c, h, w) between values [0.,1.]
    :return: tensor image of histogram (c, h, w) between values [0.,1.]
    """

    img = torch.clamp(img, min=0., max=1.)

    colors = ("red", "green", "blue")
    img_np = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    # get the dimensions
    ypixels, xpixels, bands = img_np.shape

    # get the size in inches
    dpi = plt.rcParams['figure.dpi']
    xinch = xpixels / dpi
    yinch = ypixels / dpi

    fig = plt.figure(figsize=(xinch, yinch))
    plt.xlim([-5, 260])

    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(img_np[:, :, channel_id], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.grid()
    plt.yticks(rotation=45, ha='right', fontsize=7)
    if title is not None:
        plt.title(str(title))

    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    hist_image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    hist_np = hist_image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    hist_tensor = tvtf.to_tensor(Image.fromarray(hist_np))
    plt.close(fig)

    return hist_tensor


# %% save depth tensor into rgb with colormap (instead of grayscale)

def depth_tensor_to_color_image(tensor_image, colormap='viridis'):
    cm = plt.get_cmap(colormap)

    if len(tensor_image.shape) == 4:
        tensor_image = tensor_image.squeeze()

    if len(tensor_image.shape) == 3:
        tensor_image = tensor_image[0]

    assert len(tensor_image.shape) == 2

    # color the gray scale image
    im_np = cm(tensor_image.numpy())
    depth_im_ii = torch.tensor(im_np[:, :, 0:3]).permute(2, 0, 1)

    return depth_im_ii
