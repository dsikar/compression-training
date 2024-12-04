import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionMetrics:
    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        return original_size / compressed_size
    
    @staticmethod
    def visualize_compression(original: torch.Tensor, 
                            compressed: torch.Tensor,
                            method: str,
                            save_path: str):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(original.reshape(28, 28).cpu().numpy(), cmap='gray')
        plt.title('Original')
        
        plt.subplot(122)
        # Reshape compressed data based on method
        if method == 'dct':
            size = int(np.sqrt(len(compressed)))
            reconstructed = torch.zeros(28, 28)
            reconstructed[:size, :size] = compressed.reshape(size, size)
            reconstructed = torch.fft.idctn(reconstructed, norm='ortho')
        else:
            reconstructed = compressed.reshape(16, 16)
        
        plt.imshow(reconstructed.cpu().numpy(), cmap='gray')
        plt.title(f'Compressed ({method})')
        plt.savefig(save_path)
        plt.close()

class CompressionMethods:
    _projection_matrix: Optional[torch.Tensor] = None
    _binary_mask: Optional[torch.Tensor] = None
    
    @classmethod
    def initialize_matrices(cls, input_dim: int, output_dim: int):
        try:
            if cls._projection_matrix is None:
                torch.manual_seed(42)
                cls._projection_matrix = torch.randn(input_dim, output_dim) / np.sqrt(output_dim)
                
            if cls._binary_mask is None:
                mask_size = int(np.sqrt(output_dim))
                cls._binary_mask = (torch.rand(mask_size, mask_size) > 0.5).float()
        except RuntimeError as e:
            logger.error(f"Failed to initialize compression matrices: {e}")
            raise
    
    @classmethod
    def dct_compress(cls, image: torch.Tensor) -> torch.Tensor:
        try:
            dct = torch.fft.dctn(image, norm='ortho')
            mask = torch.zeros_like(dct)
            k = 16
            mask[:k, :k] = 1
            return (dct * mask).flatten()[:k*k]
        except Exception as e:
            logger.error(f"DCT compression failed: {e}")
            raise
    
    @classmethod
    def random_projection(cls, image: torch.Tensor) -> torch.Tensor:
        try:
            if cls._projection_matrix is None:
                cls.initialize_matrices(784, 256)
            return torch.matmul(image.flatten(), cls._projection_matrix)
        except Exception as e:
            logger.error(f"Random projection failed: {e}")
            raise
    
    @classmethod
    def downsample_interpolate(cls, image: torch.Tensor) -> torch.Tensor:
        try:
            return F.interpolate(image.unsqueeze(0).unsqueeze(0),
                               size=(16, 16),
                               mode='bilinear',
                               align_corners=False).squeeze().flatten()
        except Exception as e:
            logger.error(f"Downsampling failed: {e}")
            raise
    
    @classmethod
    def binary_mask(cls, image: torch.Tensor) -> torch.Tensor:
        try:
            if cls._binary_mask is None:
                cls.initialize_matrices(784, 256)
            downsampled = F.interpolate(image.unsqueeze(0).unsqueeze(0),
                                      size=(16, 16),
                                      mode='bilinear',
                                      align_corners=False)
            return (downsampled.squeeze() * cls._binary_mask).flatten()
        except Exception as e:
            logger.error(f"Binary masking failed: {e}")
            raise

class CompressedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, compression_method: str, cache_size: int = 1000):
        self.dataset = dataset
        self.compression_method = compression_method
        self.compression_fn = getattr(CompressionMethods, f'{compression_method}_compress')
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            image, label = self.dataset[idx]
            image = image.view(28, 28)
            compressed = self.compression_fn(image)
            
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (compressed, label)
            
            return compressed, label
        except Exception as e:
            logger.error(f"Dataset access failed for index {idx}: {e}")
            raise

def train_and_evaluate(compression_method: str, 
                      epochs: int = 5) -> Dict[str, float]:
    start_time = time.time()
    results = {
        'accuracy': 0.0,
        'training_time': 0.0,
        'compression_ratio': 784/256  # Original/Compressed size
    }
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        trainset = CompressedDataset(
            datasets.MNIST('./data', train=True, download=True, transform=transform),
            compression_method
        )
        testset = CompressedDataset(
            datasets.MNIST('./data', train=False, transform=transform),
            compression_method
        )
        
        # Save visualization of first image
        original_img = trainset.dataset[0][0]
        compressed_img = trainset[0][0]
        CompressionMetrics.visualize_compression(
            original_img,
            compressed_img,
            compression_method,
            f'compression_viz_{compression_method}.png'
        )
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        
        net = CompressedNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        for epoch in range(epochs):
            net.train()
            for i, (inputs, labels) in enumerate(trainloader):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        results['accuracy'] = 100. * correct / total
        results['training_time'] = time.time() - start_time
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed for {compression_method}: {e}")
        raise

def compare_methods():
    compression_methods = ['dct', 'random_projection', 'downsample_interpolate', 'binary_mask']
    results = {}
    
    for method in compression_methods:
        logger.info(f"\nEvaluating {method} compression...")
        try:
            method_results = train_and_evaluate(method)
            results[method] = method_results
            logger.info(
                f"{method}:\n"
                f"  Accuracy: {method_results['accuracy']:.2f}%\n"
                f"  Training Time: {method_results['training_time']:.2f}s\n"
                f"  Compression Ratio: {method_results['compression_ratio']:.2f}x"
            )
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            continue
    
    return results

# Run comparison if executed directly
if __name__ == "__main__":
    compare_methods()
