import torch.utils.data

class TimeSeries0508Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
    ):
        pass

def build_ts0508_lmdb_train(
    root: str,
    seed: int = 3407,
):
    import lmdb
    import json
    import xxhash
    import scipy.io
    from pathlib import Path

    source_path = Path(root) / "train.mat"
    target_path = Path(root) / "train.lmdb"
    target_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.Environment(str(target_path), map_size=1 << 30)
    txn = env.begin(write=True)
    keys, longitudes, latitudes = [], {}, {}

    mat = scipy.io.loadmat(source_path)["C"]
    for idx, (lati, longi) in enumerate(zip(mat[0], mat[1])):
        assert lati.shape == longi.shape and lati.shape[0] == 1 and lati.ndim == 2
        lati = lati[0].tolist()
        longi = longi[0].tolist()
        mat_bytes = json.dumps({
            "latitude": lati,
            "longitude": longi,
        }).encode()
        key = xxhash.xxh32_hexdigest(bytes(target_path.resolve()) + idx.to_bytes(), seed=seed)
        keys.append(key)
        longitudes[key] = longi
        latitudes[key] = lati
        txn.put(key.encode(), mat_bytes)
    
    keys_bytes = json.dumps(keys).encode()
    txn.put(b"__keys__", keys_bytes)
    longitudes_bytes = json.dumps(longitudes).encode()
    txn.put(b"longitudes", longitudes_bytes)
    latitudes_bytes = json.dumps(latitudes).encode()
    txn.put(b"latitudes", latitudes_bytes)
    txn.commit()
    env.close()

    with (target_path / "metadata.json").open("w") as f:
        json.dump({
            "__keys__": "Keys.",
            "longitudes": "Longitudes.",
            "latitudes": "Latitudes.",
            "metadata encoding": "JSON",
            "data format": "{'longitude': ..., 'latitude': ...}",
        }, f)
    
def build_ts0508_lmdb_test(
    root: str,
    seed: int = 3407,
):
    import lmdb
    import json
    import xxhash
    import scipy.io
    from pathlib import Path

    source_path = Path(root) / "test.mat"
    target_path = Path(root) / "test.lmdb"
    target_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.Environment(str(target_path), map_size=1 << 30)
    txn = env.begin(write=True)
    keys, longitudes, latitudes, velocities, directions = [], {}, {}, {}, {}

    mat = scipy.io.loadmat(source_path)["PremS"]
    assert mat.ndim == 2 and mat.shape[0] == 1
    for idx, m in enumerate(mat.flatten()):
        assert m.ndim == 2 and m.shape[1] == 5
        longitude = m[:, 0].tolist()
        latitude = m[:, 1].tolist()
        velocity = m[:, 2].tolist()
        direction = m[:, 3].tolist()
        mat_bytes = json.dumps({
            "longitude": longitude,
            "latitude": latitude,
            "velocity": velocity,
            "direction": direction,
        }).encode()
        key = xxhash.xxh32_hexdigest(bytes(target_path.resolve()) + idx.to_bytes(), seed=seed)
        keys.append(key)
        txn.put(key.encode(), mat_bytes)
        longitudes[key] = longitude
        latitudes[key] = latitude
        velocities[key] = velocity
        directions[key] = direction
    keys_bytes = json.dumps(keys).encode()
    txn.put(b"__keys__", keys_bytes)
    longitudes_bytes = json.dumps(longitudes).encode()
    txn.put(b"longitudes", longitudes_bytes)
    latitudes_bytes = json.dumps(latitudes).encode()
    txn.put(b"latitudes", latitudes_bytes)
    velocities_bytes = json.dumps(velocities).encode()
    txn.put(b"velocities", velocities_bytes)
    directions_bytes = json.dumps(directions).encode()
    txn.put(b"directions", directions_bytes)
    txn.commit()
    env.close()

    with (target_path / "metadata.json").open("w") as f:
        json.dump({
            "__keys__": "Keys.",
            "longitudes": "Longitudes.",
            "latitudes": "Latitudes.",
            "velocities": "Velocities.",
            "directions": "Directions.",
            "metadata encoding": "JSON",
            "data format": "{'longitude': ..., 'latitude': ..., 'velocity': ..., 'direction': ...}",
        }, f)

if __name__ == "__main__":
    build_ts0508_lmdb_train("/home/af/Data/mamba-time-series")
    build_ts0508_lmdb_test("/home/af/Data/mamba-time-series")