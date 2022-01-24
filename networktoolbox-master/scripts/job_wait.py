import psutil
import time
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description='Optical Network Simulator Script')
parser.add_argument('-p', action='store', type=str, default=None, help="process name")
args = parser.parse_args()
start = time.perf_counter()
processes = [p.name() for p in psutil.process_iter()]
print(f"{bcolors.OKGREEN}waiting for {args.p} to finish{bcolors.ENDC}")
# print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
while args.p in processes:
    processes = list(filter(lambda x: True if args.p == x else False, processes))
    print("{} processes still running".format(len(processes)), end="\r")
    time_past = time.perf_counter()-start
    if time_past < 3600:
        print(f"{len(processes)} processes still running || {bcolors.BOLD}time waited: {int(time_past/60)} mins{bcolors.ENDC}", end="\r")
    elif time_past > 3600 and time_past < 3600*24:
        print("{} processes still running || time waited: {} hours".format(len(processes), round(time_past / 3600, 2)), end="\r")
    else:
        print("{} processes still running || time waited: {} days".format(len(processes), round(time_past / (3600*24), 2)), end="\r")
    time.sleep(1)
    processes = [p.name() for p in psutil.process_iter()]

