import time
import psutil
from pynvml import *
import numpy as np

import threading

class ComputerResourceMonitor:
    def __init__(self, gpu_hadle_index=0):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_hadle_index)

        self.cpu_usages = []
        self.ram_usages = []
        self.gpu_usages = []
        self.consume_time = 0
        self.consume_times = []

        self.is_monitoring = threading.Event()
        self.is_monitoring.clear()
        self.is_alive = threading.Event()
        self.is_alive.set()
    
    def start(self, time_interval=0.05):
        self.is_monitoring.set()
        while True:
            if not self.is_alive.is_set():
                break

            elif self.is_monitoring.is_set():
                # Collect CPU usage in %
                cpu_usage = psutil.cpu_percent()
                self.cpu_usages.append(cpu_usage)

                # Collect RAM usage in MB
                ram_usage = psutil.virtual_memory().used / (1024.0 ** 2)
                self.ram_usages.append(ram_usage)

                # Collect GPU usage in MB
                gpu_usage = nvmlDeviceGetMemoryInfo(self.handle).used / (1024.0 ** 2)
                self.gpu_usages.append(gpu_usage)
                time.sleep(time_interval)

            else:
                time.sleep(time_interval)

    def stop(self):
        self.consume_time = abs(self.consume_time - time.time())
        self.consume_times.append(self.consume_time)
        self.consume_time = 0
        self.is_monitoring.clear()

    def reset(self):
        self.cpu_usages.clear()
        self.ram_usages.clear()
        self.gpu_usages.clear()
        self.consume_times.clear()
        self.consume_time = 0
    
    def kill(self):
        self.is_alive.clear()
    
    def restart(self):
        self.consume_time = time.time()
        self.is_monitoring.set()
    
    def print_and_get_results(self):
        avg_cpu = np.mean(self.cpu_usages); max_cpu = np.max(self.cpu_usages)
        avg_ram = np.mean(self.ram_usages); max_ram = np.max(self.ram_usages)
        avg_gpu = np.mean(self.gpu_usages); max_gpu = np.max(self.gpu_usages)
        avg_time = np.mean(self.consume_times); max_time = np.max(self.consume_times)

        print(f"Avg CPU: {avg_cpu}%, Max CPU: {max_cpu}%")
        print(f"Avg RAM: {avg_ram}MB, Max RAM: {max_ram}MB")
        print(f"Avg GPU: {avg_gpu}MB, Max GPU: {max_gpu}MB")
        print(f"Avg time: {avg_time}sec, Max time: {max_time}sec")

        return {"avg_cpu":avg_cpu, "max_cpu":max_cpu,\
                "avg_ram":avg_ram, "max_ram":max_ram,\
                "avg_gpu":avg_gpu, "max_gpu":max_gpu,\
                "avg_time":avg_time, "max_time":max_time}
