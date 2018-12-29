#!/usr/bin/python3

import spidev
import struct
import time

class Stepper():
    def __init__(self, speed):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = speed

        self.param_to_size = [0, 3, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2]

        self.reset()
        time.sleep(0.1)

    def set_param(self, param, value):
        self.spi.xfer([param])
        for v in value:
            self.spi.xfer([value & 0xFF])

        print("Writing param: {:02X} = {:}".format(param, value))

    def get_param(self, param, read_bytes):
        self.spi.xfer([param | 0x20])
        x = []
        for i in range(read_bytes):
            x.append(self.spi.xfer([0])[0])

        ret = 0
        for i in range(read_bytes):
            ret <<= 8
            ret |= x[i]

        print("Read param {:02X} = {:}".format(param, x))
        return ret

    def dump_settings(self):
        for i in range(1, 0x19):
            ret = self.get_param(i, self.param_to_size[i])

    def run(self, speed, dir):
        print("Running stepper at {:}".format(speed))
        self.spi.xfer([0x50 | dir])
        cmd = struct.pack(">I", speed)
        cmd = cmd[1::]
        for c in cmd:
            self.spi.xfer([c])

    def stop(self):
        self.spi.xfer([0xB8])

    def set_mode(self, mode):
        print("Mode register: {:X}".format(mode))
        self.spi.xfer([0x16])
        self.spi.xfer([mode & 0xff])

    def print_status(self, status, mask, string):
        if status & mask:
            print("  {:}".format(string))

    def get_status(self):
        self.spi.xfer([0xD0])
        x = self.spi.xfer([0])
        y = self.spi.xfer([0])
        status = (x[0] << 8) + y[0]
        self.print_status(status, 0x8000, "Bit15: SCK_MOD")
        self.print_status(status, 0x1000, "Bit12: OCD")
        self.print_status(status, 0x0800, "Bit11: TH_SD")
        self.print_status(status, 0x0400, "Bit10: TH_WRN")
        self.print_status(status, 0x0200, "Bit09: UVLO")
        self.print_status(status, 0x0100, "Bit08: WRONG_CMD")
        self.print_status(status, 0x0080, "Bit07: NOTPERF_CMD")
        self.print_status(status, 0x0040, "Bit06: MOT_STATUS1")
        self.print_status(status, 0x0020, "Bit05: MOT_STATUS0")
        self.print_status(status, 0x0010, "Bit04: DIR")
        self.print_status(status, 0x0008, "Bit03: SW_EVN")
        self.print_status(status, 0x0004, "Bit02: SW_F")
        self.print_status(status, 0x0002, "Bit01: BUSY")
        self.print_status(status, 0x0001, "Bit00: HiZ")

        print("\n\n")

    def reset(self):
        self.spi.xfer([0xC0])

s = Stepper(1000000)
s.get_status()
s.dump_settings()
#s.set_mode(0x80)
s.dump_settings()
s.run(1000, 1)

spd = 0

while True:
    s.get_status()
    time.sleep(1)
    #s.run(spd, 1)
    spd += 1000

