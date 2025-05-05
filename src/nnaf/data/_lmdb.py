import lmdb

def create_imagenet100_lmdb(
    source_path: str,
    map_size: int = 1 << 40,
    commit_freq: int = 10000,
):
    import io
    import json
    from PIL import Image
    from pathlib import Path
    from alive_progress import alive_bar
    
    train_target_path = str(Path(source_path) / "train.lmdb")
    val_target_path = str(Path(source_path) / "val.lmdb")
    Path(train_target_path).mkdir(parents=True, exist_ok=True)
    Path(val_target_path).mkdir(parents=True, exist_ok=True)
    train_path_generator = Path(source_path).glob("train.*/*/*.JPEG")
    val_path_generator = Path(source_path).glob("val.*/*/*.JPEG")
    train_env = lmdb.Environment(train_target_path, map_size=map_size)
    val_env = lmdb.Environment(val_target_path, map_size=map_size)
    train_txn = train_env.begin(write=True)
    val_txn = val_env.begin(write=True)
    for path_generator, txn, env in [
        (train_path_generator, train_txn, train_env),
        (val_path_generator, val_txn, val_env),
    ]:
        written_count, keys = 0, []
        with alive_bar() as bar:
            bar.text = "Writing kv pairs..."
            for img_path in path_generator:
                key = "+".join([img_path.parent.name, img_path.name])
                with Image.open(img_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    img_data = buffer.getvalue()
                txn.put(key.encode(), img_data)
                written_count = written_count + 1
                if written_count % commit_freq == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                keys.append(key)
                bar()
            bar.text = "Writing metadata `__keys__`..."
            txn.put(b"__keys__", json.dumps(keys).encode())
            bar()
            bar.text = "Writing metadata `__len__`..."
            txn.put(b"__len__", json.dumps(len(keys)).encode())
            bar()
            bar.text = "Writing metadata `__labels__`..."
            with open(str(Path(source_path) / "Labels.json"), "rb") as f:
                labels = json.load(f)
            txn.put(b"__labels__", json.dumps(labels).encode())
            bar()
        txn.commit()
        bar.text = "All data written. This means the dataset is ready to use."
        env.close()
