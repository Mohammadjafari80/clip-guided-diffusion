########## LOSS FUNCTIONS ##########
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
    - img1 (torch.Tensor): The original image tensor.
    - img2 (torch.Tensor): The disrupted image tensor.
    - max_val (float): The maximum possible pixel value in the image (1.0 for normalized images).

    Returns:
    - psnr_value (float): The PSNR value.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # If images are identical, return infinity for PSNR
    psnr_value = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr_value

def calculate_nmi(image1, image2, num_bins=256):
    """Calculate Normalized Mutual Information (NMI) between two images."""
    hist_1 = torch.histc(image1, bins=num_bins, min=0, max=1)
    hist_2 = torch.histc(image2, bins=num_bins, min=0, max=1)

    # Normalize histograms
    hist_1 = hist_1 / hist_1.sum()
    hist_2 = hist_2 / hist_2.sum()

    # Compute joint histogram
    joint_hist = torch.histc(image1 * num_bins + image2, bins=num_bins**2, min=0, max=num_bins**2)
    joint_hist = joint_hist / joint_hist.sum()

    # Compute entropies
    H1 = -torch.sum(hist_1 * torch.log(hist_1 + 1e-10))
    H2 = -torch.sum(hist_2 * torch.log(hist_2 + 1e-10))
    H12 = -torch.sum(joint_hist * torch.log(joint_hist + 1e-10))

    # Compute Mutual Information
    I = H1 + H2 - H12

    # Normalize NMI
    nmi = (2 * I) / (H1 + H2)
    return nmi