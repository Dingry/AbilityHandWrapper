# Ability Hand Control Wrapper

This repository provides a simplified interface for interacting with the Ability Hand API, along with examples to help you get started.

## Installation

```
pip install -e .
```

## Usage Examples
To run the example script, use the following command:

```bash
python test_ability.py
```
Ensure you update the `usb_port` variable in the script to match the port to which the Ability Hand is connected on your computer and grant the sudo permission to the port.

For position control, set the `reply_mode` variable in the script to `0x11` and run the script. The default control mode is `0x21`, which is velocity control.

You can fine-tune the PID parameters in [RealAbilityHand.pd_vel_control](./ability_hand/__init__.py) to achieve better performance.

## Acknowledgements
This repository is based on the [Ability Hand API](https://github.com/psyonicinc/ability-hand-api). Thanks to Ruihan Yang, Runyu Ding, Jiyue Zhu, and Yuzhe Qin for their contributions to the development of this repository.