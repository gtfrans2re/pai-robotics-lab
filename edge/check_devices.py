"""List the OpenVINO inference devices available on this machine.

On the Meteor Lake target this should print CPU, GPU (Arc iGPU) and NPU once the
`edge` setup module has installed the iGPU compute runtime and the NPU .deb driver.
"""
import openvino as ov


def main():
    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print("Available devices:", core.available_devices)
    for dev in core.available_devices:
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception as e:  # noqa: BLE001
            name = f"(name unavailable: {e})"
        print(f"  {dev:5s}  {name}")


if __name__ == "__main__":
    main()
