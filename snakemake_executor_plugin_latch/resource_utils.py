import re
from dataclasses import dataclass
from typing import Optional

from humanfriendly import format_size, parse_size

_image_expr = re.compile(r"^(([a-zA-Z]+)://)?(?P<uri>[^/]+.*)$")


def sanitize_image_name(image: str) -> str:
    match = _image_expr.match(image)
    if match is None:
        raise ValueError(f"malformed image name: {image}")

    return match["uri"]


@dataclass
class Resources:
    cpu: int = 2000  # millicores
    mem: int = 4 * 1024**3  # bytes
    disk: int = 500 * 1024**3  # bytes
    gpus: int = 0
    gpu_type: Optional[str] = None

    def __str__(self) -> str:
        return ", ".join(
            [
                f"{self.cpu / 1000:.3g} CPUs",
                f"{format_size(self.mem, binary=True)} RAM",
                f"{format_size(self.disk, binary=True)} Storage",
                *(
                    [f"{self.gpus} {self.gpu_type} GPU(s)"]
                    if self.gpus > 0 and self.gpu_type is not None
                    else []
                ),
            ]
        )

    def __le__(self, other):
        if not isinstance(other, Resources):
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

        return self.cpu < other.cpu and self.mem < other.mem and self.disk < other.disk

    def __leq__(self, other):
        return self.__eq__(other) or self.__le__(other)

    def __ge__(self, other):
        return not self.__leq__(other)

    def __geq__(self, other):
        return not self.__le__(other)


_ng_limits = {
    "gpu-small": Resources(
        cpu=7_000,
        mem=parse_size("30 GiB"),
        disk=parse_size("1500 GiB"),
        gpus=1,
        gpu_type="nvidia-t4",
    ),
    "gpu-big": Resources(
        cpu=64_000,
        mem=parse_size("240 GiB"),
        disk=parse_size("5000 GiB"),
        gpus=1,
        gpu_type="nvidia-a10g",
    ),
    "v100-x1": Resources(
        cpu=7_000,
        mem=parse_size("48 GiB"),
        disk=parse_size("2000 GiB"),
        gpus=1,
        gpu_type="nvidia-v100",
    ),
    "v100-x4": Resources(
        cpu=30_000,
        mem=parse_size("230 GiB"),
        disk=parse_size("2000 GiB"),
        gpus=4,
        gpu_type="nvidia-v100",
    ),
    "v100-x8": Resources(
        cpu=62_000,
        mem=parse_size("400 GiB"),
        disk=parse_size("2000 GiB"),
        gpus=8,
        gpu_type="nvidia-v100",
    ),
    "cpu-32-spot": Resources(
        cpu=30_000,
        mem=parse_size("120 GiB"),
        disk=parse_size("2000 GiB"),
    ),
    "cpu-96-spot": Resources(
        cpu=94_000,
        mem=parse_size("176 GiB"),
        disk=parse_size("4949 GiB"),
    ),
    "mem-512-spot": Resources(
        cpu=62_000,
        mem=parse_size("485 GiB"),
        disk=parse_size("4949 GiB"),
    ),
    "mem-1tb": Resources(
        cpu=126_000,
        mem=parse_size("975 GiB"),
        disk=parse_size("4949 GiB"),
    ),
}


_resource_key_expr = re.compile(r"^(?P<type>mem|disk)(?:_(?P<unit>\w+))?$")


def get_resources(resources: dict[str, str]) -> Resources:
    res = Resources()

    written = {
        "cpu": False,
        "mem": False,
        "disk": False,
        "gpus": False,
        "gpu_type": False,
    }

    for key, val in resources.items():
        if not written["cpu"] and key.startswith("cpu") or key.startswith("core"):
            if isinstance(val, int):
                res.cpu = val * 1000
            elif isinstance(val, str):
                if val.endswith("m"):
                    res.cpu = parse_size(val.strip("m"))
                else:
                    res.cpu = parse_size(val) * 1000

            written["cpu"] = True
            continue

        match = _resource_key_expr.match(key)
        if match is not None:
            if written[match["type"]]:
                continue

            unit = match["unit"]
            if unit is not None:
                multiplier = parse_size(f"1 {unit}")
                in_bytes = int(val) * multiplier
            else:
                in_bytes = parse_size(val)

            setattr(res, match["type"], in_bytes)
            written[match["type"]] = True
            continue

        if not written["gpus"] and key == "gpu" or key == "gpus":
            res.gpus = int(val)
            written["gpus"] = True

        if not written["gpu_type"] and key == "gpu_type":
            res.gpu_type = val
            written["gpu_type"] = True

    return res


def validate_and_pin_gpu_resources(requests: Resources) -> Resources:
    if requests.gpus == 0:
        return requests

    assert requests.gpu_type is not None

    if requests.gpu_type == "nvidia-t4":
        if requests.gpus != 1:
            raise ValueError(
                f"Cannot request {requests.gpus} T4 GPU(s). Can only request exactly 1 T4 GPU."
            )

        return _ng_limits["gpu-small"]
    elif requests.gpu_type == "nvidia-a10g":
        if requests.gpus != 1:
            raise ValueError(
                f"Cannot request {requests.gpus} A10G GPU(s). Can only request exactly 1 A10G GPU."
            )

        return _ng_limits["gpu-big"]
    elif requests.gpu_type == "nvidia-v100":
        if requests.gpus == 1:
            return _ng_limits["v100-x1"]
        elif requests.gpus == 4:
            return _ng_limits["v100-x4"]
        elif requests.gpus == 8:
            return _ng_limits["v100-x8"]

        raise ValueError(
            f"Cannot request {requests.gpus} V100 GPU(s). Can only request exactly 1, 4, or 8 V100 GPUs."
        )

    raise ValueError(
        f"Not a valid gpu_type: {requests.gpu_type:!r}. Valid gpu_type values are 'nvidia-t4', \
        'nvidia-a10g', and 'nvidia-v100'"
    )


def validate_resource_limits(resources: Resources):
    for limits in _ng_limits.values():
        if resources <= limits:
            return

    raise ValueError(
        f"Job requests {resources} which is unsatisfiable. Maximum resources allowed are {_ng_limits['mem-1tb']}"
    )
