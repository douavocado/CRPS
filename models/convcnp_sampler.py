"""
Convolutional Conditional Neural Process with Epsilon Sampler for Probabilistic Forecasting.

This model implements a Convolutional Conditional Neural Process (ConvCNP) tailored for
spatio-temporal data, such as the 2D CFD simulations from PDEBench. It is designed to
be compatible with the `CFD2DDataset` class.

The architecture follows the principles of ConvCNPs for off-the-grid data:
1.  An encoder using a set convolution (with an RBF kernel) maps sparse context points
    onto a dense functional representation on a uniform grid.
2.  A deep convolutional neural network (a simple ResNet) processes this gridded
    representation to extract features.
3.  A decoder interpolates this feature representation back to arbitrary target
    locations using another set convolution.
4.  A final MLP, equipped with `torchnaut`'s `EpsilonSampler`, predicts multiple
    samples for the target variable, enabling probabilistic forecasting.

This approach gives the model inductive biases for permutation invariance of spatial
locations and translation equivariance on the feature grid. It handles multiple
input variables and time steps by treating them as channels in the convolutional pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnaut.crps import EpsilonSampler, crps_loss_mv
from torchnaut import crps

from common.losses import crps_loss_general, energy_score_loss

class SetConv(nn.Module):
    """
    A set convolution layer using an RBF kernel for 'off-the-grid' to 'on-the-grid' encoding.
    It maps a set of context points (coordinates and values) to a representation on a query grid.
    """
    def __init__(self, length_scale: float):
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale, dtype=torch.float32)), requires_grad=True)

    def rbf(self, dists: torch.Tensor) -> torch.Tensor:
        """Radial Basis Function kernel."""
        length_scale = torch.exp(self.log_length_scale)
        return torch.exp(-dists.pow(2) / (2 * length_scale ** 2))

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor, y_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_query: Query point coordinates, shape (B, N_query, D_coord).
            x_context: Context point coordinates, shape (B, N_context, D_coord).
            y_context: Context point values, shape (B, N_context, D_y).

        Returns:
            Concatenation of aggregated values and density, shape (B, N_query, D_y + 1).
        """
        dists = torch.cdist(x_query, x_context)  # (B, N_query, N_context)
        weights = self.rbf(dists)

        # Density channel: sum of kernels
        density = weights.sum(dim=-1, keepdim=True)

        # Normalise weights
        weights_norm = weights / (density + 1e-8)

        # Weighted sum of context values
        y_query = torch.bmm(weights_norm, y_context)

        # Concatenate density as an extra channel
        return torch.cat([y_query, density], dim=-1)


class GridInterpolator(nn.Module):
    """
    An interpolation layer using an RBF kernel for 'on-the-grid' to 'off-the-grid' decoding.
    It maps a gridded representation to values at arbitrary query locations.
    """
    def __init__(self, length_scale: float):
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale, dtype=torch.float32)), requires_grad=True)

    def rbf(self, dists: torch.Tensor) -> torch.Tensor:
        """Radial Basis Function kernel."""
        length_scale = torch.exp(self.log_length_scale)
        return torch.exp(-dists.pow(2) / (2 * length_scale ** 2))

    def forward(self, x_query: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_query: Query point coordinates, shape (B, N_query, D_coord).
            x_grid: Grid point coordinates, shape (B, N_grid, D_coord).
            y_grid: Grid point values, shape (B, N_grid, D_y).

        Returns:
            Interpolated values at query points, shape (B, N_query, D_y).
        """
        dists = torch.cdist(x_query, x_grid)
        weights = self.rbf(dists)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        y_query = torch.bmm(weights_norm, y_grid)
        return y_query


class ResConvBlock(nn.Module):
    """A residual convolutional block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ConvCNPSampler(nn.Module):
    """
    Convolutional Conditional Neural Process with an Epsilon Sampler for probabilistic output.
    """
    def __init__(self,
                 n_input_vars: int,
                 time_lag: int,
                 time_predict: int,
                 latent_dim: int = 16,
                 cnn_hidden_channels: int = 64,
                 decoder_hidden_layers: int = 2,
                 decoder_hidden_size: int = 64,
                 grid_size: int = 32,
                 length_scale: float = 0.1,
                 n_cnn_blocks: int = 4):
        super().__init__()
        self.time_predict = time_predict
        self.input_channels = n_input_vars * time_lag
        self.grid_size = grid_size

        # Create a uniform grid over [0, 1] x [0, 1]
        x_grid = torch.linspace(0, 1, grid_size)
        y_grid = torch.linspace(0, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        self.register_buffer('grid', grid)

        # 1. Encoder: off-the-grid -> on-the-grid
        self.encoder_set_conv = SetConv(length_scale=length_scale)

        # 2. CNN processor
        cnn_layers = [ResConvBlock(self.input_channels + 1, cnn_hidden_channels)]
        for _ in range(n_cnn_blocks - 1):
            cnn_layers.append(ResConvBlock(cnn_hidden_channels, cnn_hidden_channels))
        self.cnn = nn.Sequential(*cnn_layers)

        # 3. Decoder part 1: on-the-grid -> off-the-grid (interpolation)
        self.decoder_interpolator = GridInterpolator(length_scale=length_scale)

        # 4. Decoder part 2: MLP with sampler
        decoder_layers = []
        # First layer projects CNN features into the decoder's hidden space.
        decoder_layers.append(nn.Linear(cnn_hidden_channels, decoder_hidden_size))
        decoder_layers.append(nn.ReLU())

        # Inject latent sample as early as possible for probabilistic modelling.
        decoder_layers.append(EpsilonSampler(latent_dim))

        # Subsequent hidden layers process the concatenated representation.
        current_size = decoder_hidden_size + latent_dim
        # The loop runs for (decoder_hidden_layers - 1) times.
        # If decoder_hidden_layers is 1, this loop is skipped.
        for _ in range(decoder_hidden_layers - 1):
            decoder_layers.append(nn.Linear(current_size, decoder_hidden_size))
            decoder_layers.append(nn.ReLU())
            current_size = decoder_hidden_size

        # Final layer maps to the desired output size (time_predict).
        decoder_layers.append(nn.Linear(current_size, time_predict))
        self.decoder_mlp = nn.Sequential(*decoder_layers)

    def forward(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                target_coords: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Forward pass through the ConvCNP sampler.

        Args:
            input_data: Context values, shape (B, T_in, N_in, V_in).
            input_coords: Context coordinates, shape (B, N_in, 2).
            target_coords: Target coordinates, shape (B, N_out, 2).
            n_samples: Number of samples to generate.

        Returns:
            Predicted samples, shape (B, n_samples, N_out, T_out).
        """
        batch_size, _time_lag, n_input_spatial, _n_input_vars = input_data.shape
        n_target_spatial = target_coords.shape[1]

        # Reshape input_data to combine time and variable channels: (B, N_in, T_in * V_in)
        y_context = input_data.permute(0, 2, 1, 3).reshape(batch_size, n_input_spatial, -1)
        
        grid_b = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, G*G, 2)

        # 1. Encoder
        encoder_rep = self.encoder_set_conv(grid_b, input_coords, y_context) # (B, G*G, C_in+1)

        # 2. CNN: Reshape to image-like for CNN, process, and reshape back
        encoder_rep_grid = encoder_rep.permute(0, 2, 1).view(batch_size, -1, self.grid_size, self.grid_size)
        cnn_rep_grid = self.cnn(encoder_rep_grid)
        cnn_rep = cnn_rep_grid.view(batch_size, cnn_rep_grid.shape[1], -1).permute(0, 2, 1) # (B, G*G, C_cnn)

        # 3. Interpolator
        target_rep = self.decoder_interpolator(target_coords, grid_b, cnn_rep) # (B, N_out, C_cnn)

        # 4. Decoder MLP with sampling
        target_rep_flat = target_rep.view(batch_size * n_target_spatial, -1) # (B*N_out, C_cnn)

        with crps.EpsilonSampler.n_samples(n_samples):
            samples_flat = self.decoder_mlp(target_rep_flat) # (B*N_out, n_samples, T_out)

        samples = samples_flat.view(batch_size, n_target_spatial, n_samples, self.time_predict)
        return samples.permute(0, 2, 1, 3) # (B, n_samples, N_out, T_out)

    def energy_score_loss(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                  target_coords: torch.Tensor, target_data: torch.Tensor,
                  n_samples: int = 50) -> torch.Tensor:
        """
        Computes the energy score loss for the given data using flattened approach.

        Args:
            input_data: Context values, (B, T_in, N_in, V_in).
            input_coords: Context coordinates, (B, N_in, 2).
            target_coords: Target coordinates, (B, N_out, 2).
            target_data: Ground truth target values, (B, T_out, N_out).
            n_samples: Number of samples to use for energy score calculation.

        Returns:
            Scalar energy score loss.
        """
        
        samples = self.forward(input_data, input_coords, target_coords, n_samples)
        # samples shape: (B, n_samples, N_out, T_out)
        
        batch_size, _, n_target_spatial, time_predict = samples.shape
        
        # Reshape samples for energy score: (batch_size, n_samples, output_dim)
        # where output_dim = n_target_spatial * time_predict (flattened approach)
        samples_energy = samples.reshape(batch_size, n_samples, n_target_spatial * time_predict)
        
        # Reshape target data: (batch_size, output_dim)
        target_energy = target_data.permute(0, 2, 1).reshape(batch_size, n_target_spatial * time_predict)
        
        # Compute energy score loss using flattened spatial-temporal vectors
        energy_scores = energy_score_loss(samples_energy, target_energy)  # (B,)
        return energy_scores.mean()

    def crps_loss(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                  target_coords: torch.Tensor, target_data: torch.Tensor,
                  n_samples: int = 50) -> torch.Tensor:
        """
        Computes the CRPS loss for the given data.
        """
        samples = self.forward(input_data, input_coords, target_coords, n_samples)
        # samples shape: (B, n_samples, N_out, T_out)
        # target_data shape: (B, T_out, N_out)
        
        # Reshape for CRPS calculation
        # For CRPS, we need: yps [batch, num_samples, dims], y [batch, dims]
        # We'll treat each spatial location as a separate batch element
        B, n_samples, N_out, T_out = samples.shape
        
        # Reshape samples to (B * N_out, n_samples, T_out)
        samples_reshaped = samples.permute(0, 2, 1, 3).reshape(B * N_out, n_samples, T_out)
        
        # Reshape target_data to (B * N_out, T_out)
        target_reshaped = target_data.permute(0, 2, 1).reshape(B * N_out, T_out)
        
        # Compute CRPS loss
        loss = crps_loss_general(samples_reshaped, target_reshaped)
        return loss.mean()


if __name__ == '__main__':
    # Example usage of the ConvCNPSampler
    # These parameters would typically come from a config file
    B, T_in, V_in, N_in = 4, 5, 3, 100
    T_out, N_out = 3, 50
    
    # Model configuration
    model_config = {
        'n_input_vars': V_in,
        'time_lag': T_in,
        'time_predict': T_out,
        'latent_dim': 8,
        'cnn_hidden_channels': 32,
        'decoder_hidden_layers': 2,
        'decoder_hidden_size': 32,
        'grid_size': 16,
        'n_cnn_blocks': 2
    }
    model = ConvCNPSampler(**model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Create dummy data
    input_data = torch.randn(B, T_in, N_in, V_in)
    input_coords = torch.rand(B, N_in, 2)
    target_coords = torch.rand(B, N_out, 2)
    target_data = torch.randn(B, T_out, N_out)

    # Test forward pass
    print("\nTesting forward pass...")
    samples = model(input_data, input_coords, target_coords, n_samples=10)
    print(f"Input data shape: {input_data.shape}")
    print(f"Output samples shape: {samples.shape} (expected: {(B, 10, N_out, T_out)})")

    # Test CRPS loss calculation
    print("\nTesting CRPS loss...")
    loss = model.crps_loss(input_data, input_coords, target_coords, target_data, n_samples=10)
    print(f"CRPS Loss: {loss.item()}")
    
    # Check that loss computation is correct
    loss.backward()
    print("Backward pass successful.")

    # Check that some gradients are computed
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Model has gradients: {has_grad}") 