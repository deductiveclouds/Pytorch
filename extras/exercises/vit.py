
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dimension:int=768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dimension, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f'Image size: {image_resolution} is not divisible by Patch size: {self.patch_size}.'
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)
    
class VIT(nn.Module):
    def __init__(self, img_size:int=224, num_channels:int=3, patch_size:int=16, embedding_dim:int=768, dropout:float=0.1, mlp_size:int=3072, num_transformer_layers:int=12, num_heads:int=12, num_classes:int=1000):
        super().__init__()
        assert img_size % patch_size == 0, f'Image size: {img_size} must be divisible by Patch size: {patch_size}.'
        self.patch_embedding = PatchEmbedding(in_channels=num_channels, patch_size=patch_size, embedding_dimension=embedding_dim)
        self.class_token = nn.Parameter(data=torch.randn(size=(1, 1, embedding_dim)), requires_grad=True)
        num_patches = (img_size**2) // (patch_size**2)
        self.positional_embedding = nn.Parameter(data=torch.randn(size=(1, num_patches+1, embedding_dim)))
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=mlp_size, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=num_transformer_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), nn.Linear(in_features=embedding_dim, out_features=num_classes))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat(tensors=(class_token, x), dim=1)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.mlp_head(x[:, 0])
        return x
