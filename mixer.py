#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import docker
import json
import logging
import os
import sortedcontainers
import shutil
import tempfile
import threading
import time

from captor import Captor


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')


class Report:
    def __init__(self, path):
        self.path = path.replace('.report.json', '')
        self.name = os.path.split(path)[1]
        prefix = 'predictions-'
        suffix = '.csv.report.json'
        self.short = self.name.replace(prefix, '').replace(suffix, '')
        with open(path, 'rb') as handle:
            report = json.loads(handle.read().decode('utf-8'))
            self.concordance = report['concordance']
            self.originality = report['originality']
            self.consistency = report['consistency']
            self.logloss = report['logloss']

    @staticmethod
    def compare(x, y):
        if not x.concordance and y.concordance: return 1
        if x.concordance and not y.concordance: return -1
        if not x.originality and y.originality: return 1
        if x.originality and not y.originality: return -1
        if x.consistency < y.consistency: return 1
        if x.consistency > y.consistency: return -1
        if x.logloss > y.logloss: return 1
        if x.logloss < y.logloss: return -1
        return 0

    def __lt__(self, other): return Report.compare(self, other) < 0
    def __le__(self, other): return Report.compare(self, other) <= 0
    def __eq__(self, other): return Report.compare(self, other) == 0
    def __ne__(self, other): return Report.compare(self, other) != 0
    def __gt__(self, other): return Report.compare(self, other) > 0
    def __ge__(self, other): return Report.compare(self, other) >= 0

    def __str__(self):
        template = '{:.5f},{:3d},{},{},{}'
        logloss = self.logloss
        consistency = int(self.consistency)
        concordance = '+' if self.concordance else '-'
        originality = '+' if self.originality else '-'
        short = self.short
        return template.format(
            logloss, consistency, concordance, originality, short)


class Volume:
    @staticmethod
    def locate():
        client = docker.APIClient(base_url='unix://var/run/docker.sock')
        host = os.environ['HOSTNAME']
        for volume in client.inspect_container(host)['Mounts']:
            if volume['Destination'] != '/var/run/docker.sock':
                Volume.source_path = volume['Source']
                Volume.destination_path = volume['Destination']

    @staticmethod
    def source():
        try:
            return Volume.source_path
        except Exception:
            Volume.locate()
            return Volume.source_path

    @staticmethod
    def destination():
        try:
            return Volume.destination_path
        except Exception:
            Volume.locate()
            return Volume.destination_path

    @staticmethod
    def path(name):
        return os.path.join(Volume.destination(), name)


class Mixer(threading.Thread):
    def __init__(self):
        super(Mixer, self).__init__()
        self.logger = logging.getLogger('mixer')
        self.logger.setLevel(logging.DEBUG)
        self.expect()
        self.captor = Captor(Volume.destination(), '*.report.json')
        self.reports = sortedcontainers.SortedList()
        self.lut = {}
        self.start()

    def expect(self):
        expecting = os.getenv('EXPECTING', None)
        if expecting is not None and expecting != '':
            self.expecting = expecting.split(':')
        else:
            self.expecting = []
        self.logger.info('Expecting models: {}'.format(str(self.expecting)))

    def ready(self):
        for expected in self.expecting:
            if not os.path.isfile(expected):
                self.logger.info('Missing models: {}'.format(expected))
                return False
        return True

    def collect(self, requested):
        while not self.captor.empty():
            path = self.captor.grab()
            report = Report(path)
            short = report.short
            if short not in self.lut:
                self.lut[short] = report
                self.reports.add(report)
            if short in requested:
                requested.remove(short)

    def encode(self, decoded):
        return '-'.join(sorted(list(decoded)))

    def decode(self, encoded):
        return set(encoded.split('-'))

    def ensemble(self, short):
        client = docker.APIClient(base_url='unix://var/run/docker.sock')
        ensemble = list(self.decode(short))
        ensemble = [self.lut[item].path for item in ensemble]
        predicting = os.path.join(
            Volume.destination(),
            'predictions-{}.csv'.format(short))
        environment = {
            'OPERATION': 'Ensemble',
            'ENSEMBLING': ':'.join(ensemble),
            'PREDICTING': predicting
        }
        config = client.create_host_config(
            binds=['{}:/data'.format(Volume.source())])
        container = client.create_container(
            'r606020/numerai-jf', detach=True, environment=environment,
            volumes=['/data'], host_config=config)
        client.start(container=container.get('Id'))

    def ranking(self):
        if 0 == len(self.reports):
            return
        handle, name = tempfile.mkstemp()
        header = 'LOGLOSS,CONSISTENCY,CONCORDANCE,ORIGINALITY\n'
        os.write(handle, header.encode('utf-8'))
        for report in self.reports:
            os.write(handle, '{}\n'.format(report).encode('utf-8'))
        os.close(handle)
        shutil.copy(name, os.path.join(Volume.destination(), 'ranking.csv'))
        os.remove(name)

    def usable(self, item):
        concordant = item.concordance
        original = item.originality
        consistent = item.consistency >= 75
        return concordant or original or consistent

    def mix(self, requested):
        usable = [item for item in self.reports if self.usable(item)]
        length = len(usable)
        if length < 2:
            return
        for i in range(length - 1):
            a = self.decode(usable[i].short)
            for j in range(i + 1, length):
                b = self.decode(usable[j].short)
                c = a.union(b)
                short = self.encode(c)
                if short not in self.lut and short not in requested:
                    self.ensemble(short)
                    requested.add(short)
                    return

    def run(self):
        self.logger.info('Mixer ready')
        requested = set()
        then = 0
        try:
            while True:
                self.collect(requested)
                self.ranking()
                backoff = 10 if self.ready() else 60 * 60
                now = time.time()
                if then + backoff < now and not requested:
                    self.mix(requested)
                    then = now
                time.sleep(10)
        except Exception:
            self.logger.exception('Mixer failed')


if __name__ == '__main__':
    Mixer()
