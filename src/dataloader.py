from torch.utils.data import DataLoader
from src.dataset import (
    SkinLesionDataset,
    train_transforms,
    val_test_transforms
)

def get_dataloaders(
    train_csv  : str = 'outputs/train.csv',
    val_csv    : str = 'outputs/val.csv',
    test_csv   : str = 'outputs/test.csv',
    task       : str = 'multiclass',
    batch_size : int = 32,
    num_workers: int = 0,
):
    train_dataset = SkinLesionDataset(train_csv, train_transforms,   task)
    val_dataset   = SkinLesionDataset(val_csv,   val_test_transforms, task)
    test_dataset  = SkinLesionDataset(test_csv,  val_test_transforms, task)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader