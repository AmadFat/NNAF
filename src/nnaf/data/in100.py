from typing import Literal

def build_in100_tfrec(source_path: str):
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # keep it as it was
    import json
    import random
    import xxhash
    import shutil
    import tensorflow as tf
    from pathlib import Path
    from subprocess import call
    from alive_progress import alive_bar

    ideal_shard_size = 1 << 27

    # create hash label
    with (Path(source_path) / "Labels.json").open("rb") as f:
        label2ann = json.load(f)
    label2hash, hash2ann = dict(), dict()
    for label, ann in label2ann.items():
        label_hashed = xxhash.xxh32_hexdigest(label.encode())
        while label_hashed in hash2ann:
            label_hashed = xxhash.xxh32_hexdigest(label_hashed.encode() + b"oh-my-hash")
        label2hash[label] = label_hashed
        hash2ann[label_hashed] = ann
    assert len(label2hash) == len(hash2ann) == 100

    tfrec_path = Path(source_path) / "tfrec"
    shutil.rmtree(tfrec_path, ignore_errors=True)
    tfrec_path.mkdir(parents=True, exist_ok=True)
    with (tfrec_path / "labels.json").open("w") as f:
        json.dump(hash2ann, f)

    for part in ["train", "val"]:
        shard_idx = 1
        image_paths = list(Path(source_path).glob(f"{part}.*/*/*.JPEG"))
        random.shuffle(image_paths)
        target_path = tfrec_path / part
        shutil.rmtree(target_path, ignore_errors=True)
        target_path.mkdir(parents=True, exist_ok=True)
        shard_path = target_path / f"{shard_idx:04d}.tfrecord"
        writer = tf.io.TFRecordWriter(path=str(shard_path)) # no compression

        with alive_bar() as bar:
            for path in image_paths:
                if shard_path.stat().st_size > ideal_shard_size:
                    writer.close()
                    call(["tfrecord2idx", str(shard_path), str(shard_path.with_suffix(".idx"))])
                    shard_idx += 1
                    shard_path = target_path / f"{shard_idx:04d}.tfrecord"
                    writer = tf.io.TFRecordWriter(path=str(shard_path))
                img_bytes = path.read_bytes()
                label = label2hash[path.parent.name].encode()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image/jpeg": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                    "label/bytes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                })).SerializeToString()
                writer.write(example)
                bar()
            writer.close()
            call(["tfrecord2idx", str(shard_path), str(shard_path.with_suffix(".idx"))])

def imagenet100_loader(
    source_path: str,
    part: Literal["train", "val"] = "train",
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    seed: int = 3407,
    use_dali: bool = True,
):
    if use_dali:
        from nvidia import dali
        from pathlib import Path
        from nvidia.dali.auto_aug import rand_augment
        from nvidia.dali import pipeline_def

        target_path = Path(source_path) / "tfrec" / part
        tfrec_paths = list(target_path.glob("*.tfrecord"))
        idx_paths = list(target_path.glob("*.idx"))

        # dali.pipeline.Pipeline()

        @pipeline_def(enable_conditionals=True, exec_dynamic=True, exec_async=True, exec_pipelined=False)
        def my_ppl(tfrec_paths, idx_paths):
            inputs = dali.fn.readers.tfrecord(
                path=tfrec_paths,
                index_path=idx_paths,
                features={
                    "image/jpeg": dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                    "label/bytes": dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                },
            )
            images = dali.fn.decoders.image(
                inputs["image/jpeg"],
                device="mixed",
                output_type=dali.types.DALIImageType.RGB,
                # cache_size=1 << 10, # 1 GB
                cache_threshold=0, # 128 KB
                # cache_type="largest",
                # use_fast_idct=False,
            ).gpu()
            images = rand_augment.rand_augment(images, n=2, m=9)
            images = dali.fn.resize(
                images,
                device="gpu",
                size=(224, 224),
                interp_type=dali.types.DALIInterpType.INTERP_LINEAR,
            )
            return images, inputs["label/bytes"]

        pipe = my_ppl(
            tfrec_paths=tfrec_paths,
            idx_paths=idx_paths,
            batch_size=batch_size,
            num_threads=num_workers,
            prefetch_queue_depth=2,
            seed=seed,
        )
        pipe.build()
        return pipe

    else:
        import torch
        from PIL import Image
        from pathlib import Path
        class Dataset(torch.utils.data.Dataset):
            def __init__(
                self,
                source_path: str,
                part: str,
                transform=None,
            ):
                self.img_paths = list(Path(source_path).glob(f"{part}.*/*/*.JPEG"))
                self.transform = transform
            
            def __len__(self):
                return len(self.img_paths)
            
            def __getitem__(self, idx):
                img_path = self.img_paths[idx]
                img = Image.open(img_path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
        
        from torchvision.transforms import v2
        transform = v2.Compose([
            v2.ToImage(),
            v2.RandAugment(2, 9),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, True),
        ])

        data = Dataset(source_path, part, transform)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        return loader

def io_benchmark_fetch_data(args):
    from alive_progress import alive_bar

    batch_size = 64
    test_num = int(3e4 / batch_size)
    
    if args == "dali": 

        pipe = imagenet100_loader(
            "/home/af/Data/ImageNet100",
            part="train",
            batch_size=64,
            num_workers=32,
        )
        with alive_bar() as bar:
            while True:
                images, labels = pipe.run()
                bar()

    elif args == "torch":
        loader = imagenet100_loader(
            "/home/af/Data/ImageNet100",
            part="train",
            batch_size=64,
            num_workers=32,
            use_dali=False
        )
        with alive_bar() as bar:
            for i, images in enumerate(loader):
                print(images.shape)
                bar()