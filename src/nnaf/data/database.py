from collections.abc import Iterator
# import sqlite3, lmdb, xxhash
from nvidia import nvimgcodec
from typing import Literal
from pathlib import Path

def create_cv_database(
    target_path: str,
    map_size: int = 1 << 40,
):
    from pathlib import Path
    import lmdb, sqlite3
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    sql_conn = sqlite3.connect(target_path / "keys.db")
    sql_cursor = sql_conn.cursor()
    sql_cursor.execute("""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
)
    """)
    sql_cursor.execute("""
CREATE TABLE IF NOT EXISTS main (
    key BLOB PRIMARY KEY,
    label BLOB,
    meta TEXT
)
    """)
    sql_cursor.execute("PRAGMA journal_mode=WAL;")
    sql_cursor.execute("PRAGMA synchronous=NORMAL;")
    sql_cursor.execute("PRAGMA cache_size=-64000;")
    sql_cursor.execute("PRAGMA temp_store=MEMORY;")
    sql_cursor.execute("PRAGMA mmap_size=268435456;")
    sql_conn.commit()
    lmdb_env = lmdb.Environment(str(target_path), map_size=map_size, writemap=True)
    return sql_conn, sql_cursor, lmdb_env

def write_cv_batch(
    sql_cursor: sqlite3.Cursor,
    lmdb_txn: lmdb.Transaction,
    path_batch: list[str],
    label_batch: list[str],
    image_format: Literal[".png", ".jpg"],
    quality: int,
    label_hashing_dict: dict[str, bytes],
    seed=3407,
    bar=None,
):
    enc = nvimgcodec.Encoder()
    dec = nvimgcodec.Decoder()
    img_batch = dec.read(path_batch)
    img_bytes_batch = enc.encode(
        img_batch,
        image_format,
        params=nvimgcodec.EncodeParams(
            quality=quality,
            jpeg_encode_params=nvimgcodec.JpegEncodeParams(optimized_huffman=True),
        ),
    )
    for path, img_bytes, label in zip(path_batch, img_bytes_batch, label_batch):
        if bar:
            bar()
        path = Path(path)
        key_hashed = xxhash.xxh32_digest(str(path.absolute()).encode(), seed=seed)
        while lmdb_txn.get(key_hashed) is not None:
            key_hashed = xxhash.xxh32_digest(key_hashed + b"oh-my-hash", seed=seed)
        lmdb_txn.put(key_hashed, img_bytes)
        if label in label_hashing_dict.keys():
            label_hashed = label_hashing_dict[label]
        else:
            label_hashed = xxhash.xxh32_digest(label.encode(), seed=seed)
            while label_hashed in label_hashing_dict.values():
                label_hashed = xxhash.xxh32_digest(label_hashed + b"oh-my-hash", seed=seed)
            label_hashing_dict[label] = label_hashed
        meta = ""
        sql_cursor.execute("INSERT OR REPLACE INTO main (key, label, meta) VALUES (?, ?, ?)", (key_hashed, label_hashed, meta))


def decode_bytes_benchmark(option):
    from .in100 import ImageNet100Dataset
    data = ImageNet100Dataset(
        root="/home/af/Data/ImageNet100",
        try_fastdb=True,
    )
    test_size = int(1e4)
    from alive_progress import alive_bar
    if option == "pywebp":
        import webp
        import torch
        with alive_bar(test_size) as bar:
            for i in range(test_size):
                img_bytes, _ = data[i]
                img = webp.WebPData.from_buffer(img_bytes)
                img = img.decode(webp.WebPColorMode.RGB)
                img = torch.from_numpy(img)
                bar()
    if option == "pil":
        from PIL import Image
        import io
        from torchvision.transforms.v2 import functional as F
        with alive_bar(test_size) as bar:
            for i in range(test_size):
                img_bytes, _ = data[i]
                img = Image.open(io.BytesIO(img_bytes))
                img = img.convert("RGB")
                img = F.to_image(img)
                bar()
