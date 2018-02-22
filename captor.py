#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
import glob
import logging
import os
import pyinotify
import queue
import sys
import threading


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')


class Captor(pyinotify.ProcessEvent, threading.Thread):
    def __init__(self, directory, pattern):
        threading.Thread.__init__(self)
        self.directory = directory
        self.pattern = pattern
        self.spec = os.path.join(self.directory, self.pattern)
        self.logger = logging.getLogger('captor')
        self.logger.setLevel(logging.DEBUG)
        wm = pyinotify.WatchManager()
        mask = pyinotify.IN_CLOSE_NOWRITE | pyinotify.IN_CLOSE_WRITE
        wm.add_watch(self.directory, mask)
        self.notifier = pyinotify.Notifier(wm, self)
        self.queue = queue.Queue()
        self.collect()
        self.daemon = True
        self.start()

    def collect(self):
        for found in glob.glob(os.path.join(self.directory, self.pattern)):
            self.logger.info('Collected ({}): {}'.format(self.spec, found))
            self.queue.put(found)

    def run(self):
        self.logger.info('Captor (%s) ready', self.spec)
        try:
            self.notifier.loop()
        except Exception:
            self.logger.exception('Captor ({}) failed'.format(self.spec))

    def process_IN_CLOSE_WRITE(self, event):
        directory, name = os.path.split(event.pathname)
        if fnmatch.fnmatch(name, self.pattern):
            self.logger.info('Captured ({}): {}'.format(self.spec, name))
            self.queue.put(event.pathname)

    def empty(self):
        return self.queue.empty()

    def grab(self):
        return self.queue.get(block=True, timeout=None)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: captor.py DIRECTORY PATTERN')
    else:
        Captor(sys.argv[1], sys.argv[2]).join()
