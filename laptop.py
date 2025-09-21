import os
import platform
import subprocess


def get_cpu_score():
    if platform.system() == "Windows":
        cpu_info = subprocess.check_output("wmic cpu get NumberOfCores, MaxClockSpeed", shell=True).decode()
        lines = cpu_info.strip().split("\n")
        cores, speed = map(int, lines[1].split())
    else:
        cpu_info = subprocess.check_output("lscpu | grep -E 'Core|MHz'", shell=True).decode()
        cores = int(subprocess.check_output("nproc", shell=True).decode().strip())
        speed = int(float(cpu_info.split("\n")[1].split(":")[1].strip()))

    score = min(4, (cores / 4) + (speed / 3000))
    return round(score, 1)


def get_ram_score():
    if platform.system() == "Windows":
        ram_info = subprocess.check_output("wmic memorychip get Capacity", shell=True).decode()
        total_ram = sum(int(x) for x in ram_info.split()[1:]) // (1024 ** 3)
    else:
        ram_info = subprocess.check_output("free -m", shell=True).decode()
        total_ram = int(ram_info.split("\n")[1].split()[1]) // 1024

    score = min(2, total_ram / 8)
    return round(score, 1)


def get_ssd_score():
    if platform.system() == "Windows":
        ssd_info = subprocess.check_output("winsat disk -seq -ran", shell=True).decode()
        speed = int(ssd_info.split("Disk Sequential 64.0 Read ")[1].split(" MB/s")[0])
    else:
        ssd_info = subprocess.check_output("sudo hdparm -Tt /dev/nvme0n1", shell=True).decode()
        speed = int(float(ssd_info.split("Timing buffered disk reads:")[1].split(" MB/sec")[0]))

    score = min(2, speed / 2000)
    return round(score, 1)


def get_gpu_score():
    if platform.system() == "Windows":
        gpu_info = subprocess.check_output("wmic path win32_videocontroller get Name", shell=True).decode()
    else:
        gpu_info = subprocess.check_output("lspci | grep -i VGA", shell=True).decode()

    score = 2 if "NVIDIA" in gpu_info or "AMD" in gpu_info else 1
    return round(score, 1)


# Tính tổng điểm
cpu_score = get_cpu_score()
ram_score = get_ram_score()
ssd_score = get_ssd_score()
gpu_score = get_gpu_score()
total_score = cpu_score + ram_score + ssd_score + gpu_score

print(f"🔹 CPU: {cpu_score}/4")
print(f"🔹 RAM: {ram_score}/2")
print(f"🔹 SSD: {ssd_score}/2")
print(f"🔹 GPU: {gpu_score}/2")
print(f"🔥 Tổng điểm: {total_score}/10")
