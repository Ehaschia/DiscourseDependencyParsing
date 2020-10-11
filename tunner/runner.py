import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from multiprocessing import Pool
import paramiko
import time
from collections import deque

from tunner.utils import ROOT_DIR

THREAD = 18
script_path = ROOT_DIR + '/scripts/'
cli_prefix = 'sh ' + script_path
skip = {}


def runner(idx, cli):
    client = paramiko.SSHClient()
    client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    print("Begin to connect " + str(idx))
    client.connect('node' + str(idx))
    print("Connected " + str(idx))
    stdin, stdout, stderr = client.exec_command(cli_prefix + cli)
    print("Run " + cli + " on " + str(idx))
    # detect done or not
    while True:
        time.sleep(60)
        detector = paramiko.SSHClient()
        detector.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        print("Begin to detect " + str(idx))
        detector.connect('node' + str(idx))
        stdin, stdout, stderr = detector.exec_command('top -bn1 | grep hanwj')
        programs = stdout.read().decode().strip().split('\n')
        detector.close()
        # find python
        python_line = None
        for line in programs:
            if line.find('python') != -1:
                python_line = line

        # if not python line
        if python_line is None:
            client.close()
            break
        else:
            cpu = float([line for line in python_line.split(' ') if len(line) > 0][8])
            if cpu < 100.0:
                client.close()
                break

    print("Finish " + cli + " on " + str(idx))
    return idx, cli


def multiprocess(configs, thread):
    thread = min(len(configs), thread)
    pool = Pool(processes=thread)
    runnings = {}
    finished = []
    # init threads
    for i in range(1, thread + 1 + len(skip)):
        time.sleep(3)
        if i in skip:
            continue
        runnings[i] = pool.apply_async(runner, (i, configs.pop()))

    while len(finished) != len(configs):
        time.sleep(60)
        for idx in runnings.keys():
            if runnings[idx] is not None:
                if runnings[idx].ready():
                    finished_thread = runnings[idx]
                    runnings[idx] = None
                    idx, cli = finished_thread.get()
                    finished.append(cli)
                    if len(configs) > 0:
                        runnings[idx] = pool.apply_async(runner, (idx, configs.pop()))
            else:
                pass


if __name__ == '__main__':
    # freeze_support()
    configs = deque(os.listdir(script_path))
    print(len(configs))
    multiprocess(configs, THREAD)
