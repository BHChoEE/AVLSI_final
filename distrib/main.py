import tensorflow as tf
import threading
import multiprocessing
import numpy as np
import os
import shutil
from model import Model


################ GLOBAL VARIABLE ####################
N_WORKERS = multiprocessing.cpu_count()


class Worker(object):
    def __init__(self, name, global_model):
        pass
    def work(self):
        pass

def main():
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        GLOBAL_MODEL = Model()
        workers = []
        # create worker
        for i in range(N_WORKERS):
            i_name = 'W%i'%i
            workers.append(Worker(i_name, GLOBAL_MODEL))
    
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())


    ## multi-thread manipulation
    worker_threads = []
    for worker in workers:
        job = lambda : worker.work()
        t = threading.Thread(target = job)
        t.start()
        worker_threads.append(t)

    ## sync all the workers
    COORD.join(worker_threads)

if __name__ == '__main__':
    main()
