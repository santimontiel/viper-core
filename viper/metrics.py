from typing import Optional

import torch
from torch import Tensor
from torchmetrics import Metric
    

class MaskedMAE(Metric):
    """
    Mean Absolute Error over a user-provided mask.
    """
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("num_valid", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ):
        # Compute masked absolute error
        abs_error = torch.abs(preds - target)
        masked_error = abs_error[mask]

        self.sum_abs_error += masked_error.sum()
        self.num_valid += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum_abs_error / self.num_valid


class MaskedRMSE(Metric):
    """
    Root Mean Squared Error over a user-provided mask.
    """
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("num_valid", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ):
        # Compute masked squared error
        sq_error = (preds - target) ** 2
        masked_sq_error = sq_error[mask]

        self.sum_squared_error += masked_sq_error.sum()
        self.num_valid += mask.sum()

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.sum_squared_error / self.num_valid)


class DiceScore(Metric):
    """
    Dice Score metric for segmentation tasks.
    
    The Dice Score (also known as F1 score or Dice coefficient) measures the 
    overlap between predicted and ground truth segmentation masks.
    
    Formula: Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        num_classes: Number of classes in the segmentation task
        ignore_index: Optional class index to ignore in metric computation
        average: Method to average scores across classes. Options:
            - 'micro': Calculate metrics globally across all classes
            - 'macro': Calculate metrics for each class and find their unweighted mean
            - 'weighted': Calculate metrics for each class and find their weighted mean
            - 'none': Return scores for each class separately
        smooth: Smoothing constant to avoid division by zero (default: 1e-6)
        kwargs: Additional keyword arguments for the base Metric class
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        average: str = 'macro',
        smooth: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average
        self.smooth = smooth
        
        if average not in ['micro', 'macro', 'weighted', 'none']:
            raise ValueError(
                f"Average must be one of ['micro', 'macro', 'weighted', 'none'], got {average}"
            )
        
        # Register buffers for accumulating statistics
        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("pred_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update state with predictions and targets.
        
        Args:
            preds: Predictions from model. Shape: (B, C, H, W) for logits or (B, H, W) for class indices
            target: Ground truth labels. Shape: (B, H, W)
        """
        # Handle logits input - convert to class predictions
        if preds.ndim == 4:  # (B, C, H, W)
            preds = torch.argmax(preds, dim=1)  # (B, H, W)
        
        # Flatten predictions and targets
        preds = preds.flatten()
        target = target.flatten()
        
        # Create mask for valid indices (excluding ignore_index)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            preds = preds[valid_mask]
            target = target[valid_mask]
        
        # Convert to one-hot encoding
        preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes)  # (N, C)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)  # (N, C)
        
        # Calculate intersection and sums for each class
        intersection = (preds_one_hot * target_one_hot).sum(dim=0).float()  # (C,)
        pred_sum = preds_one_hot.sum(dim=0).float()  # (C,)
        target_sum = target_one_hot.sum(dim=0).float()  # (C,)
        
        # Update states
        self.intersection += intersection
        self.pred_sum += pred_sum
        self.target_sum += target_sum
        self.total_samples += preds.numel()
    
    def compute(self) -> Tensor:
        """
        Compute the Dice Score based on accumulated state.
        
        Returns:
            Dice score(s) as a tensor
        """
        # Calculate Dice score per class
        dice_per_class = (2 * self.intersection + self.smooth) / (
            self.pred_sum + self.target_sum + self.smooth
        )
        
        if self.average == 'none':
            return dice_per_class
        
        elif self.average == 'micro':
            # Global calculation
            total_intersection = self.intersection.sum()
            total_pred = self.pred_sum.sum()
            total_target = self.target_sum.sum()
            return (2 * total_intersection + self.smooth) / (total_pred + total_target + self.smooth)
        
        elif self.average == 'macro':
            # Unweighted mean across classes
            # Only average over classes that appear in the target
            valid_classes = self.target_sum > 0
            if valid_classes.sum() > 0:
                return dice_per_class[valid_classes].mean()
            else:
                return torch.tensor(0.0, device=dice_per_class.device)
        
        elif self.average == 'weighted':
            # Weighted mean by class frequency
            weights = self.target_sum / self.target_sum.sum()
            return (dice_per_class * weights).sum()
        
        return dice_per_class


# Example usage
if __name__ == "__main__":
    # Create metric instance
    metric = DiceScore(
        num_classes=5,
        ignore_index=255,  # Commonly used for unlabeled pixels
        average='macro'
    )
    
    # Simulate predictions and targets
    batch_size, height, width = 4, 128, 128
    num_classes = 5
    
    # Logits from model (B, C, H, W)
    preds_logits = torch.randn(batch_size, num_classes, height, width)
    
    # Ground truth (B, H, W)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    # Add some ignore_index pixels
    target[target == 0] = 255
    
    # Update metric
    metric.update(preds_logits, target)
    
    # Compute final score
    dice_score = metric.compute()
    print(f"Dice Score: {dice_score:.4f}")
    
    # Reset for next epoch
    metric.reset()
    
    # Example with per-class scores
    metric_per_class = DiceScore(num_classes=5, ignore_index=255, average='none')
    metric_per_class.update(preds_logits, target)
    per_class_scores = metric_per_class.compute()
    print(f"Per-class Dice Scores: {per_class_scores}")