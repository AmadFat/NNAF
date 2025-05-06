from collections.abc import Callable
import lmdb.tool
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

@profile
def build_in100_jxl_lmdb(
    source_path: str,
    map_size: int = 1 << 40,
    commit_freq: int = 10000,
    num_producers: int = 32,
    quality: int = 85,
    seed: int = 0
):
    import lmdb
    import queue
    import random
    import xxhash
    import threading
    import pillow_jxl
    from pathlib import Path
    from alive_progress import alive_bar
    random.seed(seed)
    consumer_lock = threading.Lock()
    producer_lock = threading.Lock()

    for part in ["train", "val"]:

        DONE = "DONE"
        assign_queue = queue.Queue(maxsize=1024)
        product_queue = queue.Queue(maxsize=1024)
        written, keys = 0, []
        num_producers_active = num_producers

        path_gen = Path(source_path).glob(f"{part}.*/*/*.JPEG")
        target_path = Path(source_path) / f"{part}-jpegxl.lmdb"
        target_path.mkdir(parents=True, exist_ok=True)
        env = lmdb.Environment(str(target_path), map_size=map_size)
        txn = env.begin(write=True)

        @profile
        def assigner():
            for path in path_gen:
                while True:
                    try:
                        assign_queue.put(path, timeout=1)
                        break
                    except queue.Full:
                        continue
            for _ in range(num_producers):
                assign_queue.put(DONE)
            print("Assigner completed")
        
        @profile
        def producer():
            nonlocal target_path, map_size
            env = lmdb.Environment(str(target_path), map_size=map_size)
            txn = env.begin(write=False)
            while True:
                path = assign_queue.get(timeout=1)
                if path is DONE:
                    assign_queue.task_done()
                    break
                key = xxhash.xxh32_hexdigest(str(path.resolve()).encode(), seed=seed).encode()
                while txn.get(key) is not None:
                    key = xxhash.xxh32_hexdigest(key + b"oh-my-hash", seed=seed).encode()
                label = path.parent.name
                buf = io.BytesIO()
                img = Image.open(str(path))
                img.save(buf, format="JXL", lossless_jpeg=False, quality=quality)
                img_bytes = buf.getvalue()
                buf.close()
                while True:
                    try:
                        product_queue.put((key, label, img_bytes), timeout=1)
                        break
                    except queue.Full:
                        continue
                assign_queue.task_done()
            with producer_lock:
                nonlocal num_producers_active
                num_producers_active = num_producers_active - 1
                if num_producers_active == 0:
                    while True:
                        try:
                            product_queue.put(DONE, timeout=1)
                            break
                        except queue.Full:
                            continue

        @profile
        def consumer():
            nonlocal written, keys, bar, env, txn
            while True:
                if num_producers_active == 0 and product_queue.empty():
                    break
                item = product_queue.get(timeout=1)
                if item is DONE:
                    product_queue.task_done()
                    break
                key, label, img_bytes = item
                txn.put(key, img_bytes)
                with consumer_lock:
                    written = written + 1
                    keys.append({
                        "key": key.decode(),
                        "label": label,
                    })
                if written % commit_freq == 0:
                    txn = txn.commit()
                    txn = env.begin(write=True)
                bar()
                product_queue.task_done()

        with alive_bar() as bar:
            assigner_thread = threading.Thread(target=assigner)
            assigner_thread.daemon = True
            assigner_thread.start()

            producer_threads = []
            for _ in range(num_producers):
                t = threading.Thread(target=producer)
                t.daemon = True
                producer_threads.append(t)
                t.start()

            consumer_thread = threading.Thread(target=consumer)
            consumer_thread.daemon = True
            consumer_thread.start()
            
            assigner_thread.join()
            print("All assigners finished")

            for t in producer_threads:
                t.join()
            print("All producers finished")

            consumer_thread.join()
            print("All consumers finished")

        print(f"Writing metadata for {part}...")
        txn.put(b"__keys__", json.dumps(keys).encode())
        txn.put(b"__len__", json.dumps(len(keys)).encode())
        with open(Path(source_path) / "Labels.json", "r") as f:
            labels = json.load(f)
        txn.put(b"__labels__", json.dumps(labels).encode())

        txn.commit()
        env.sync()
        env.close()
        print(f"Completed {part} dataset: {written} images")
