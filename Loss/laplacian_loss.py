
import torch
import torch.nn.functional as F

def laplacian_filter(tensor):
    """Applies a Laplacian filter to a tensor."""
    laplacian_kernel = torch.tensor([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]], dtype=torch.float32,device='cuda')

    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(tensor.shape[1], 1, 1, 1)

    # Handling the case for a batch of images
    padding = 1
    stride = 1
    filtered_tensor = F.conv2d(tensor, laplacian_kernel, padding=padding, stride=stride, groups=tensor.shape[1])
    return filtered_tensor


def resize_image(image_tensor, new_size=(128, 128)):
    """
    Resize the image tensor to a new size.
    :param image_tensor: Input image tensor.
    :param new_size: New size (height, width) as a tuple.
    :return: Resized image tensor.
    """
    # Resize image using bilinear interpolation
    resized_image = F.interpolate(image_tensor, size=new_size, mode='bilinear', align_corners=False)
    return resized_image

def normalize_image(image_tensor):
    image_tensor=image_tensor/255
    return (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

def laplacian_edge_loss_per_batch(output, label, reduction_ratio=0.5):
    """
    Computes the Laplacian edge loss for each item in the batch between the output and the label
    after resizing and normalization.
    :param output: Output tensor from the model, including batch dimension.
    :param label: Ground truth tensor, including batch dimension.
    :param reduction_ratio: Ratio to reduce the size of the images.
    :return: Laplacian edge loss averaged over the batch.
    """
    batch_losses = []
    for i in range(output.size(0)):  # Loop over the batch dimension
        # Normalize the images to [0, 1]
        output_normalized = normalize_image(output[i])
        label_normalized = normalize_image(label[i])

        # Calculate new size based on the reduction ratio
        original_size = output_normalized.size()[1:]  # Get only H and W dimensions
        new_size = (int(original_size[0] * reduction_ratio), int(original_size[1] * reduction_ratio))

        # Resize images
        output_resized = resize_image(output_normalized.unsqueeze(0), new_size)
        label_resized = resize_image(label_normalized.unsqueeze(0), new_size)

        # Apply Laplacian filter to both output and label tensors
        output_edges = laplacian_filter(output_resized)
        label_edges = laplacian_filter(label_resized)

        # Compute the mean squared error loss between the edge maps for the current batch item
        loss = F.mse_loss(output_edges, label_edges)
        batch_losses.append(loss)

    # Compute the average loss over the batch
    average_loss = torch.mean(torch.stack(batch_losses))
    return average_loss
