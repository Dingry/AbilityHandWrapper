from ability_hand.hand_control import RealAbilityHand
import time
import numpy as np

# Control mode:
# 0x10: Position control
# 0x20: Velocity control

# 0x10: Reply finger position, current, touch sensors
# 0x11: Reply finger position, rotor velocity, touch sensor
# 0x12: Reply finger position, current, rotor velocity


hand = RealAbilityHand(
    usb_port="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
    reply_mode=0x21,
    hand_address=0x50,
    plot_touch=False,
    verbose=True,
)

hand.start_process()

t_1 = time.time()

while time.time() - t_1 < 5:
    joint_angle = [15, 15, 15, 15, 15, -15]
    joint_angle = list(np.array(joint_angle) / 180 * np.pi)

    hand.set_joint_angle(joint_angle, reply_mode=0x21)


t_2 = time.time()

data = {"hand_state": [], "joint_angle": []}

while time.time() - t_2 < 5:
    time.sleep(0.1)
    joint_angle = [80, 80, 80, 80, 80, -50]
    joint_angle = list(np.array(joint_angle) / 180 * np.pi)

    hand.set_joint_angle(joint_angle)


hand.stop_process()
