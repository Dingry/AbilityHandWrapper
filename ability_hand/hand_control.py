import multiprocessing as mp
import time
from typing import Optional
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
import numpy as np

# import keyboard
from .python.PPP_stuffing import *
from .python.abh_api_core import *
from .python.finger_4bar.abh_finger_4bar import (
    get_abh_4bar_driven_angle,
)


class RealAbilityHand:
    def __init__(
        self,
        baud=460800,
        usb_port="/dev/ttyUSB0",
        reply_mode=0x10,
        hand_address=0x50,
        plot_touch=False,
        verbose=False,
    ) -> None:
        self.usb_port = usb_port
        self.hand_address = hand_address
        self.baud = baud

        # Serial variables
        self.ser: Optional[serial.Serial] = None
        self.setup_serial(self.baud)

        self.reset_count = 0
        self.stuff_data = False
        self.isRS485 = False

        self.reset_count = 0
        self.stuff_data = False
        self.isRS485 = False

        self.num_writes = 0
        self.num_reads = 0

        self.control_frequency = 80

        self.bytebuffer = bytes([])
        self.stuff_buffer = np.array([])

        self.reply_mode = mp.Value("i", reply_mode)
        self.farr_to_barrs = {
            0x10: farr_to_dposition,
            0x20: farr_to_dvelocity,
            0x30: farr_to_dcurrent,
        }

        self.plot_touch = plot_touch

        self.prev_error = 0
        self.prev_integral = 0
        self.idx = 1

        self.verbose = verbose

        # Shared memory
        self.current_joint_pos_read = mp.Array("d", 6, lock=True)
        self.current_joint_vel_read = mp.Array("d", 6, lock=True)
        self.current_touch_read = mp.Array("d", 30, lock=True)
        self.current_joint_target: Optional[mp.Array] = None

        # Multiprocess variable
        self.is_running = mp.Value("b", False)
        self.control_process: Optional[mp.Process] = None
        self.plot_process: Optional[mp.Process] = None
        self.tstart = None

    def setup_reply_mode(self, reply_mode):
        with self.reply_mode.get_lock():
            self.reply_mode.value = reply_mode

    def __del__(self):
        self.stop_process()

    def start_process(self):
        if self.current_joint_target is None:
            raw_pos = self.get_hand_state()["raw_pos"]
            self.current_joint_target = mp.Array("d", 6, lock=True)
            with self.current_joint_target.get_lock():
                self.current_joint_target[:] = raw_pos

        self.control_process = mp.Process(target=self.control_function)
        self.tstart = time.perf_counter()
        self.is_running.value = True
        self.control_process.start()
        if self.plot_touch:
            self.plot_process = mp.Process(target=self.live_plot)
            self.plot_process.start()

        if self.verbose:
            print(f"[AbilityHand {self.usb_port}] start process.")

    def stop_process(
        self,
    ):
        with self.is_running.get_lock():
            self.is_running.value = False
        if self.control_process is not None:
            self.control_process.join()
            if self.plot_touch:
                self.plot_process.join()
        if self.verbose:
            print(f"[AbilityHand {self.usb_port}] stop process.")

    def setup_serial(self, baud):
        self.ser = serial.Serial(self.usb_port, baud, timeout=0, write_timeout=0)
        assert self.ser is not None
        print(f"[AbilityHand {self.usb_port}] connected!")
        self.ser.reset_input_buffer()

    @staticmethod
    def create_misc_msg(cmd):
        barr = [(struct.pack("<B", 0x50))[0], (struct.pack("<B", cmd))[0]]
        barr_sum = 0
        for b in barr:
            barr_sum = barr_sum + b
        chksum = (-barr_sum) & 0xFF
        barr.append(chksum)
        return barr

    @staticmethod
    def pd_vel_control(current_pos, dest_pos, prev_error, prev_integral, dt=0.01):
        Kp = np.array([1, 1, 1, 1, 0.3, 0.3]) * 1.0  # Proportional gain
        Kd = Kp / 50
        Ki = np.array([0.003, 0.003, 0.003, 0.003, 0.001, 0.001]) * 0.0  # Integral gain

        error = dest_pos - current_pos
        derivative = (error - prev_error) / dt
        integral = np.clip(prev_integral + error * dt, -100, 100)
        output = Kp * error + Kd * derivative + Ki * integral
        output = np.clip(output, -0.2, 0.2)

        return output, error, integral

    # Generate Message to send to hand from array of farr (floating point)
    def joint_to_cmd_msg(self, farr):
        # global reply_mode, hand_address
        # farr could be: position / velocity / current
        with self.reply_mode.get_lock():
            # print('real reply mode', self.reply_mode.value, self.farr_to_barrs[self.reply_mode.value & 0xF0])
            msg = self.farr_to_barrs[self.reply_mode.value & 0xF0](
                self.hand_address, farr, self.reply_mode.value - 0x10
            )
        return msg

    @staticmethod
    def joint_remap_10_qpos_to_6(joint):
        # Ability Hand Real Order:
        # index, middle, ring, pinky, thumb l2, thumb l1

        # URDF Order:
        # thumb l1, thumb l2, index * 2, middle * 2, ring * 2, pinky * 2

        return joint[[2, 4, 6, 8, 1, 0]]

    @staticmethod
    def joint_remap_6_qpos_to_10(joint):
        augmented_joint_pos_read = []
        augmented_joint_pos_read += list(joint[-2:][::-1])
        for i in range(0, 4):
            current_temp_read = joint[i]
            augmented_joint_pos_read += [
                current_temp_read,
                get_abh_4bar_driven_angle(current_temp_read),
            ]
        return np.array(augmented_joint_pos_read)

    def set_joint_angle(self, joint_angle, reply_mode=None, dt=0.01):
        if len(joint_angle) == 10:
            joint_angle = self.joint_remap_10_qpos_to_6(joint_angle)
        if reply_mode is not None:
            self.setup_reply_mode(reply_mode)

        if self.current_joint_target is None:
            self.current_joint_target = mp.Array("d", 6, lock=True)

        with self.reply_mode.get_lock():
            reply_mode = self.reply_mode.value
        if reply_mode & 0x10 == 0x10:
            with self.current_joint_target.get_lock():
                self.current_joint_target[:] = joint_angle[:]
        elif reply_mode & 0x20 == 0x20:
            with self.current_joint_pos_read.get_lock():
                current_joint_pos = np.empty(6)
                current_joint_pos[:] = self.current_joint_pos_read

            vel, self.prev_error, self.prev_integral = self.pd_vel_control(
                current_joint_pos / 180 * np.pi,
                joint_angle.copy(),
                self.prev_error,
                self.prev_integral,
                dt=dt,
            )

            with self.current_joint_target.get_lock():
                self.current_joint_target[:] = vel[:]
        else:
            raise NotImplementedError

    def _inner_set_joint_angle(self, joint_angle):
        # Reindex and Convert Unit
        if isinstance(joint_angle, list):
            joint_angle = np.array(joint_angle)
        if len(joint_angle) == 10:
            joint_angle = self.joint_remap_10_qpos_to_6(joint_angle)
        joint_angle = joint_angle / (2 * np.pi) * 360

        joint_angle = list(joint_angle).copy()

        msg = self.joint_to_cmd_msg(joint_angle)
        # We use Stuff Data = True, isRS485 = False for now, for other variant, check ability API
        # print("before write")
        # print(joint_angle, self.reply_mode.value, msg)
        self.ser.write(PPP_stuff(bytearray(msg)))
        # print("after write")

        # Read first response byte - format header
        # data = self.ser.read(1)
        # while (len(nb) == 0):
        #     nb = self.ser.read(512)  # gigantic read size with nonblocking

        # Note that we only read once to avoid hanging
        nb = self.ser.read(512)  # gigantic read size with nonblocking
        # print("length nb: ", len(nb))

        self.bytebuffer = self.bytebuffer + nb

        # Read hand state
        if len(self.bytebuffer) != 0:  # redundant, but fine to keep
            npbytes = np.frombuffer(self.bytebuffer, np.uint8)
            for b in npbytes:
                payload, self.stuff_buffer = unstuff_PPP_stream(b, self.stuff_buffer)
                if len(payload) != 0:
                    rPos, rI, rV, rFSR = parse_hand_data(payload)
                    if (rPos.size + rI.size + rV.size + rFSR.size) != 0:
                        """If the parser got something, print it out. This is blocking, time consuming, and execution time is not guaranteed, but it is guaranteed to reduce average bandwidth"""
                        with self.current_joint_pos_read.get_lock():
                            self.current_joint_pos_read[:] = rPos[:]
                        if len(rV) == 6:
                            with self.current_joint_vel_read.get_lock():
                                self.current_joint_vel_read[:] = rV[:]
                        with self.current_touch_read.get_lock():
                            self.current_touch_read[:] = rFSR[:]

                        self.bytebuffer = bytes([])
                        self.stuff_buffer = np.array([])

    def get_hand_state(self):
        # Filling Missing Values // Ability Read Angle -> Convert to Radian
        with self.current_joint_pos_read.get_lock():
            current_qpos = np.empty(6)
            current_qpos[:] = self.current_joint_pos_read[:]

        with self.current_joint_vel_read.get_lock():
            current_qvel = np.empty(6)
            current_qvel[:] = self.current_joint_vel_read[:]

        with self.current_touch_read.get_lock():
            current_touch = np.empty(30)
            current_touch[:] = self.current_touch_read[:]

        temp_current_joint_read = current_qpos / 180 * np.pi
        augmented_joint_pos_read = self.joint_remap_6_qpos_to_10(
            temp_current_joint_read
        )

        return {
            "pos": augmented_joint_pos_read,
            "raw_pos": temp_current_joint_read,
            "vel": current_qvel,
            "touch": current_touch,
        }

    def control_function(self):
        while True:
            with self.is_running.get_lock():
                if not self.is_running.value:
                    break
            with self.current_joint_target.get_lock():
                current_target_joint_pos = np.empty(6)
                current_target_joint_pos[:] = self.current_joint_target[:]
            self._inner_set_joint_angle(current_target_joint_pos)
            # hand_state = self.get_hand_state()
            time.sleep(1 / self.control_frequency)

    def get_hand_state_wrapper(self):
        while 1:
            t = time.time() - self.tstart
            touch_list = self.get_hand_state()["touch"]
            if isinstance(touch_list, np.ndarray):
                touch_list = touch_list.tolist()
            touch_list.insert(0, t)
            yield touch_list

    def live_plot(self):
        global fig, ax, lines, xbuf, ybuf, num_lines, bufwidth, tstart, x_data, y_data, expname

        expname = "default"
        fig, ax = plt.subplots()
        plt.setp(ax, ylim=(0, 4500))  # manually set axis y limits
        plt.setp(ax, xlim=(0, 30))
        plt.title("Touch Sensor Data")
        plt.xlabel("Time(s)")
        plt.ylabel("Raw Touch Data")

        num_lines = 30
        bufwidth = 500

        lines = []
        xbuf = []
        ybuf = []
        x_data = []
        y_data = []
        for i in range(num_lines):
            lines.append(ax.plot([], [])[0])
            xbuf.append([])
            ybuf.append([])
            y_data.append([])
        # initalize all xy buffers to 0
        for i in range(0, num_lines):
            y_data[i].append(0)
            for j in range(0, bufwidth):
                xbuf[i].append(0)
                ybuf[i].append(0)
        x_data.append(0)

        tstart = self.tstart

        print("animation start")

        anim = animation.FuncAnimation(
            fig,
            self.animate,
            init_func=self.init,
            frames=self.get_hand_state_wrapper(),
            interval=0,
            blit=True,
            save_count=50,
        )

        print("animation end")
        plt.show()

    def init(
        self,
    ):  # required for blitting to give a clean slate.
        global lines

        for line in lines:
            line.set_data([], [])
        return lines

    def animate(self, args):
        global ax, lines, xbuf, ybuf, num_lines, bufwidth, x_data, y_data, expname
        t_interval = 10
        for i in range(0, num_lines):
            del xbuf[i][0]
            del ybuf[i][0]
            xbuf[i].append(args[0])
        x_data.append(args[0])
        i = 0
        for arg in args:
            if i > 0:
                ybuf[i - 1].append(arg)
                y_data[i - 1].append(arg)
            i = i + 1
        for i, line in enumerate(lines):
            line.set_data(xbuf[i], ybuf[i])

        xmin = min(xbuf[0])
        xmax = max(xbuf[0])

        # print(type(xbuf))
        if time.time() - tstart >= self.idx * t_interval:
            x_data = []
            y_data = []
            for i in range(num_lines):
                y_data.append([])
            self.idx += 1

        plt.setp(ax, xlim=(xmin, xmax))
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)
        return lines

    def get_data(self):
        data = self.get_hand_state()
        data = np.concatenate([data["raw_pos"], data["vel"], data["touch"]])
        return data

    def save_data(self, path, time):
        data = self.get_hand_state()
        data = np.concatenate([data["raw_pos"], data["vel"], data["touch"]])
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{time}.npy"
        np.save(path, data)
