from collections.abc import Callable
import torch.utils.data
from PIL import Image
import json
import io

class ImageNet100Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        import lmdb
        from pathlib import Path
        db_path = str(Path(root) / ("train.lmdb" if train else "val.lmdb"))
        self.env = lmdb.Environment(db_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.keys = json.loads(txn.get(b"__keys__"))
            self.labels = json.loads(txn.get(b"__labels__"))
            self.label2identity = {
                label: {
                    "id": label_id,
                    "name": name,
                } for label_id, (label, name) in enumerate(self.labels.items())
            }
        assert len(self.keys) == self.__len__()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        with self.env.begin() as txn:
            return json.loads(txn.get(b"__len__"))

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = self.keys[idx]
            img = Image.open(io.BytesIO(txn.get(key.encode()))).convert("RGB")
            ann = self.label2identity[key.split("+")[0]]["id"]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            ann = self.target_transform(ann)
        return img, ann

def ImageNet100Loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn: Callable = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    in_order: bool = False,
):
    def default_collate(batch):
        imgs, anns = zip(*batch)
        imgs = torch.stack(imgs)
        anns = torch.as_tensor(anns)
        return imgs, anns

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn or default_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        in_order=in_order,
    )
    return loader
