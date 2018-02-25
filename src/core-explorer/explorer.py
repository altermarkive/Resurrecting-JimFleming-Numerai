#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import docker
import logging
import luigi
import os
import time
import traceback
import zipfile

from luigi.contrib.docker_runner import DockerTask
from numerapi.numerapi import NumerAPI


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')


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


class Downloading(luigi.Task):
    def requires(self):
        return []

    def run(self):
        napi = NumerAPI()
        while True:
            try:
                napi.download_current_dataset(
                    dest_filename=self.output().path, unzip=False)
                break
            except Exception as exception:
                time.sleep(10)
                continue

    def output(self):
        return luigi.LocalTarget(Volume.path('data.zip'))


class Extracting(luigi.Task):
    def requires(self):
        return [Downloading()]

    def run(self):
        path = self.input()[0].path
        directory = Volume.destination()
        handle = zipfile.ZipFile(path)
        for name in handle.namelist():
            handle.extract(name, directory)
        handle.close()
        os.remove(path)
        with self.output().open('wb') as handle:
            handle.write('done')

    def output(self):
        return luigi.LocalTarget(Volume.path('extracting.done'))


class Computing(DockerTask):
    retry_count = 3
    command = None
    container_options = {
        'detach': True
    }
    binds = [
        '{}:/data'.format(Volume.source())
    ]
    docker_url = 'unix://var/run/docker.sock'
    auto_remove = True


class RunningAS(Computing):
    image = 'r606020/numerai-as'
    name = 'as_{}'.format(int(time.time()))
    environment = {
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-as.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-as.csv'))


class RunningAW(Computing):
    image = 'r606020/numerai-aw'
    name = 'aw_{}'.format(int(time.time()))
    environment = {
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-aw.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-aw.csv'))


class RunningSVM(Computing):
    image = 'r606020/numerai-ay'
    name = 'svm_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'SVM',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-svm.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-svm.csv'))


class RunningWND(Computing):
    image = 'r606020/numerai-jw'
    name = 'wnd_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'WideAndDeep',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-wnd.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-wnd.csv'))


class RunningXGB(Computing):
    image = 'r606020/numerai-jw'
    name = 'xgb_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'XGBoost',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-xgb.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-xgb.csv'))


class Preparing(Computing):
    image = 'r606020/numerai-jf'
    name = 'preparation_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'PrepareData',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv')
    }

    def requires(self):
        return [Extracting()]

    def output(self):
        return luigi.LocalTarget(Volume.path('test_data.csv'))


class RunningTSNE2D(Computing):
    image = 'r606020/numerai-jf'
    name = 'tsne2d_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'tSNE2D',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'STORING': Volume.destination()
    }

    def requires(self):
        return [Preparing()]

    def output(self):
        return luigi.LocalTarget(Volume.path('tsne_2d_50p_poly.npz'))


class RunningTSNE(Computing):
    image = 'r606020/numerai-jf'
    name = 'tsne_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'tSNESummary',
        'TSNE_2D_ONLY': '1',
        'STORING': Volume.destination()
    }

    def requires(self):
        return [RunningTSNE2D()]

    def output(self):
        return luigi.LocalTarget(Volume.path('tsne.npz'))


class RunningTFNN(Computing):
    image = 'r606020/numerai-jf'
    name = 'tfnn_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'TFNN',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-tfnn.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-tfnn.csv'))


class RunningTFPAIR(Computing):
    image = 'r606020/numerai-jf'
    name = 'tfpair_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'TFPairwise',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-tfpair.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-tfpair.csv'))


class RunningLR(Computing):
    image = 'r606020/numerai-jf'
    name = 'lr_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'AdditionalLogisticRegression',
        'TSNE_2D_ONLY': '1',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-lr.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-lr.csv'))


class RunningPAIR(Computing):
    image = 'r606020/numerai-jf'
    name = 'pair_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'Pairwise',
        'TSNE_2D_ONLY': '1',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-pair.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-pair.csv'))


class RunningFM(Computing):
    image = 'r606020/numerai-jf'
    name = 'fm_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'FactorizationMachines',
        'TSNE_2D_ONLY': '1',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-fm.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-fm.csv'))


class RunningGBT(Computing):
    image = 'r606020/numerai-jf'
    name = 'gbt_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'GradientBoostingTrees',
        'TSNE_2D_ONLY': '1',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-gbt.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-gbt.csv'))


class RunningTPOT(Computing):
    image = 'r606020/numerai-jf'
    name = 'tpot_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'TPOT',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-tpot.csv')
    }

    def requires(self):
        return [RunningTSNE()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-tpot.csv'))


class RunningTPOTLONG(Computing):
    image = 'r606020/numerai-jf'
    name = 'tpotlong_{}'.format(int(time.time()))
    environment = {
        'OPERATION': 'TPOT',
        'TIME_LIMIT_ALL': '4320',
        'TIME_LIMIT_PART': '20',
        'PREPARED_TRAINING': Volume.path('train_data.csv'),
        'PREPARED_VALIDATING': Volume.path('valid_data.csv'),
        'PREPARED_TESTING': Volume.path('test_data.csv'),
        'PREDICTING': Volume.path('predictions-tpotlong.csv')
    }

    def requires(self):
        return [RunningTPOT()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-tpotlong.csv'))


class RunningASLONG(Computing):
    image = 'r606020/numerai-as'
    name = 'aslong_{}'.format(int(time.time()))
    environment = {
        'TIME_LIMIT_ALL': '259200',
        'TIME_LIMIT_PART': '1200',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-aslong.csv')
    }

    def requires(self):
        return [RunningAS()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-aslong.csv'))


class RunningAWLONG(Computing):
    image = 'r606020/numerai-aw'
    name = 'awlong_{}'.format(int(time.time()))
    environment = {
        'TIME_LIMIT': '4320',
        'TRAINING': Volume.path('numerai_training_data.csv'),
        'TESTING': Volume.path('numerai_tournament_data.csv'),
        'PREDICTING': Volume.path('predictions-awlong.csv')
    }

    def requires(self):
        return [RunningAW()]

    def output(self):
        return luigi.LocalTarget(Volume.path('predictions-awlong.csv'))


class Exploring(luigi.WrapperTask):
    def requires(self):
        yield RunningSVM()
        yield RunningWND()
        yield RunningXGB()
        yield RunningTFNN()
        yield RunningTFPAIR()
        yield RunningLR()
        yield RunningPAIR()
        yield RunningFM()
        yield RunningGBT()
        yield RunningASLONG()
        yield RunningAWLONG()
        yield RunningTPOTLONG()


if __name__ == '__main__':
    luigi.run()
