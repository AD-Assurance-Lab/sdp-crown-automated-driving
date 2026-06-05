import torch
import torch.nn as nn

class SemanticPerturbationLayer(nn.Module):
    def __init__(self, nominal_image, spatial_mask=False):
        super(SemanticPerturbationLayer, self).__init__()
        self.spatial_mask = spatial_mask
        # Force the nominal image into a continuous memory block
        self.register_buffer('x_0', nominal_image.contiguous())

        # Generate the spatial Mask M (1 for road, 0 for sky)
        _, c, h, w = nominal_image.shape
        mask = torch.zeros((1, c, h, w), device=nominal_image.device)
        mask[:, :, h//2:, :] = 1.0  # Bottom half = 1.0
        
        # .contiguous() prevents auto_LiRPA .view() crashes on 3-channel images
        self.register_buffer('M', mask.contiguous())
        self.register_buffer('nominal_sky', (nominal_image * (1.0 - mask)).contiguous())

    def forward(self, eps):
        eps_c = eps[:, 0].view(-1, 1, 1, 1) 
        eps_b = eps[:, 1].view(-1, 1, 1, 1)
        
        # Calculate the affine transformation
        transformed_tensor = self.x_0 * (1.0 + eps_c) + eps_b
        
        if self.spatial_mask:
            # Add the pre-computed clear sky to the transformed road
            degraded_tensor = self.nominal_sky + (transformed_tensor * self.M)
        else:
            # Global weather (Fog/Night)
            degraded_tensor = transformed_tensor
            
        # Ensure the final output tensor is contiguous before auto_LiRPA processes it
        return torch.clamp(degraded_tensor, 0.0, 1.0).contiguous()

class SemanticVerifiedNetwork(nn.Module):
    def __init__(self, base_model, nominal_image, condition_name=""):
        super(SemanticVerifiedNetwork, self).__init__()
        
        # Automatically toggle spatial masking based on the weather condition
        use_mask = condition_name.lower() in ["snow", "rain"]
        self.semantic_layer = SemanticPerturbationLayer(nominal_image, spatial_mask=use_mask)
        self.base_model = base_model

    def forward(self, eps):
        degraded_image = self.semantic_layer(eps)
        return self.base_model(degraded_image)
