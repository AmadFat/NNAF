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
        self.env = lmdb.Environment(
            db_path,
            readonly=True,
            lock=False,
            readahead=False,
            create=False,
        )
        self.txn = self.env.begin(write=False)
        self.keys = json.loads(self.txn.get(b"__keys__"))
        self.labels = json.loads(self.txn.get(b"__labels__"))
        self.key2label = json.loads(self.txn.get(b"__key2label__"))
        self.label2id = {l: i for i, (l, _) in enumerate(self.labels.items())}
        self.transform = transform
        self.target_transform = target_transform

        import atexit
        atexit.register(self.close)
    
    def close(self):
        self.env.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        buf = io.BytesIO(self.txn.get(key.encode()))
        img = Image.open(buf).convert("RGB")
        buf.close()
        ann = self.label2id[self.key2label[key]]
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
    pin_memory: bool = False,
    in_order: bool = False,
):
    def default_collate(batch):
        imgs, anns = zip(*batch)
        imgs = torch.stack(imgs)
        anns = torch.as_tensor(anns)
        return imgs, anns
    import multiprocessing
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn or default_collate,
        num_workers=num_workers or multiprocessing.cpu_count(),
        pin_memory=pin_memory,
        in_order=in_order,
    )
    return loader

def build_in100_lmdb(
    source_path: str,
    map_size: int = 1 << 40,
    commit_freq: int = 1000,
    num_consumers: int = 0,
    quality: int = 100,
    seed: int = 3407,
):
    import lmdb
    import queue
    import random
    import xxhash
    import multiprocessing
    from pathlib import Path
    from alive_progress import alive_bar
    
    # Auto-configure optimal worker count
    if num_consumers <= 0:
        num_consumers = max(1, multiprocessing.cpu_count())
    
    random.seed(seed)

    for part in ["train", "val"]:
        path_list = list(Path(source_path).glob(f"{part}.*/*/*.JPEG"))
        target_path = Path(source_path) / f"{part}.lmdb"
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Optimize LMDB parameters for performance
        env = lmdb.Environment(
            str(target_path), 
            map_size=map_size,
        )
        
        random.shuffle(path_list)
        
        # Increase queue sizes for better throughput
        img_queue = multiprocessing.Queue(256)
        bytes_queue = multiprocessing.Queue(256)
        keys, key2label = [], {}
        
        # Producer process - better for I/O operations
        def producer(path_list, iq):
            for path in path_list:
                try:
                    img = Image.open(str(path)).convert("RGB")
                    iq.put((path, img), block=True)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
            for _ in range(num_consumers):
                iq.put(None)  # Signal end

        # Consumer process - CPU-bound task benefits from multiprocessing
        def consumer(iq, bq):
            while True:
                item = iq.get()
                if item is None:
                    break
                
                path, img = item
                try:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", lossless=False, Q=quality)
                    img_bytes = buf.getvalue()
                    buf.close()
                    img.close()  # Release memory
                    
                    
                    bq.put((path, img_bytes, path.parent.name))
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
            
            bq.put(None)  # Signal completion

        # Start producer process
        producer_proc = multiprocessing.Process(
            target=producer, 
            args=(path_list, img_queue)
        )
        producer_proc.start()
        
        # Start consumer processes
        consumers = []
        for _ in range(num_consumers):
            p = multiprocessing.Process(
                target=consumer, 
                args=(img_queue, bytes_queue)
            )
            p.start()
            consumers.append(p)
        
        # Process results and write to LMDB in batches
        txn = env.begin(write=True)
        total_written = 0
        completed_consumers = 0
        batch = []
        
        with alive_bar(len(path_list)) as bar:
            while completed_consumers < num_consumers:
                try:
                    item = bytes_queue.get()
                    if item is None:
                        completed_consumers += 1
                        continue
                    
                    path, img_bytes, label = item
                    key = xxhash.xxh32_hexdigest(str(path.resolve()).encode(), seed=seed).encode()
                    # Check for key collisions
                    while txn.get(key) is not None:
                        key = xxhash.xxh32_hexdigest(key + b"oh-my-hash", seed=seed).encode()
                    
                    # Add to batch
                    batch.append((key, img_bytes, label))
                    
                    # Write batch when it reaches sufficient size
                    if len(batch) >= 100:  # Batch writes for better performance
                        for k, v, l in batch:
                            txn.put(k, v)
                            k = k.decode()
                            keys.append(k)
                            key2label[k] = l
                        batch = []
                        total_written += 100
                        bar.text = f"length of img_queue: {img_queue.qsize()}; length of bytes_queue: {bytes_queue.qsize()}"
                        bar(100)
                        
                        # Commit transaction periodically
                        if total_written % commit_freq == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                except queue.Empty:
                    # Check if all processes are done
                    if not producer_proc.is_alive() and all(not p.is_alive() for p in consumers):
                        break
        
        # Write remaining batch
        if batch:
            for k, v, l in batch:
                txn.put(k, v)
                k = k.decode()
                keys.append(k)
                key2label[k] = l
            bar(len(batch))
        
        txn.put(b"__keys__", json.dumps(keys).encode())
        txn.put(b"__key2label__", json.dumps(key2label).encode())
        with (Path(source_path) / "Labels.json").open() as f:
            labels = json.load(f)
        txn.put(b"__labels__", json.dumps(labels).encode())

        # Clean up
        txn.commit()
        env.sync()
        env.close()

        with Path(target_path / "metadata.json").open("w") as f:
            json.dump({
                "__keys__": "All keys in the dataset",
                "__key2label__": "Mapping from keys to labels",
                "__labels__": "Mapping from label IDs to human-readable names",
                "image encoding": "JPEG",
                "metadata encoding": "JSON",
            }, f)
        
        # Ensure all processes are terminated
        for p in consumers:
            p.join()
        producer_proc.join()
        
        print(f"Completed {part} dataset: {len(path_list)} images")
