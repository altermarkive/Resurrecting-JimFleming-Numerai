#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import requests
import threading
import time
import sys

from captor import Captor
from numerapi.numerapi import NumerAPI


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')


class Evaluator(threading.Thread):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.logger = logging.getLogger('evaluator')
        self.logger.setLevel(logging.DEBUG)
        self.login()

    def login(self):
        public_id = os.environ['PUBLIC_ID']
        private_secret = os.environ['PRIVATE_SECRET']
        while True:
            try:
                self.napi = NumerAPI(public_id, private_secret)
                break
            except Exception:
                self.logger.exception('Login failed')
                time.sleep(10)

    def upload(self, prediction):
        while True:
            try:
                self.logger.info('Uploading prediction: {}'.format(prediction))
                self.napi.upload_predictions(file_path=prediction)
                self.logger.info('Uploaded prediction: {}'.format(prediction))
                break
            except requests.exceptions.HTTPError as error:
                if error.response.status_code == 429:
                    self.logger.info('Backing off')
                    time.sleep(30 * 60)
                else:
                    self.logger.exception('Network failure')
                    time.sleep(60)
            except Exception as exception:
                self.logger.exception('Upload failure')
                time.sleep(10)

    def check(self, prediction):
        while True:
            try:
                self.logger.info('Checking submission: {}'.format(prediction))
                status = self.napi.submission_status()
                self.logger.info('Got {}: {}'.format(prediction, str(status)))
                logloss_ready = status['validation_logloss'] is not None
                concordance_ready = not status['concordance']['pending']
                originality_ready = not status['originality']['pending']
                if logloss_ready and concordance_ready and originality_ready:
                    return status
                else:
                    time.sleep(10)
            except Exception:
                self.logger.exception('Checking submission failed')
                time.sleep(10)

    def report(self, prediction, status):
        result = {
            'logloss': status['validation_logloss'],
            'consistency': status['consistency'],
            'concordance': status['concordance']['value'],
            'originality': status['originality']['value']
        }
        with open(prediction + '.report.json', 'wb') as handle:
            handle.write(json.dumps(result).encode('utf-8'))

    def submit(self, prediction):
        self.upload(prediction)
        time.sleep(5)
        self.report(prediction, self.check(prediction))

    def run(self):
        captor = Captor(os.getenv('STORING'), 'predictions*.csv')
        while True:
            while not captor.empty():
                prediction = captor.grab()
                if not os.path.isfile(prediction + '.report.json'):
                    self.submit(prediction)
            time.sleep(1)


if __name__ == '__main__':
    evaluator = Evaluator()
    if len(sys.argv) < 2:
        evaluator.start()
    else:
        for prediction in sys.argv[1:]:
            evaluator.submit(prediction)
