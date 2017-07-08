#!/usr/bin/env python
import os.path
import sys

import cv2
from model.local_video_searcher import LocalSearcher
from model.features import GistGenerator
from model.predictor import KFlannPredictor
import unittest
import model
import numpy as np
import fake_filesystem
import fake_tempfile
import sys
import random

class TestSaveModel(unittest.TestCase):
    def setUp(self):
        self.real_tempfile = sys.modules['tempfile']
        self.real_os = sys.modules['os']
        self.filesystem = fake_filesystem.FakeFilesystem()
        sys.modules['tempfile'] = fake_tempfile.FakeTempfileModule(
            self.filesystem)
        sys.modules['os'] = fake_filesystem.FakeOsModule(self.filesystem)
        random.seed(32452)
        

    def tearDown(self):
        sys.modules['tempfile'] = self.real_tempfile
        sys.modules['os'] = self.real_os

    def createRandomImage(self, shape=(256,256,3)):
        image = np.array(256*np.random.random(shape),
                         dtype=np.uint8)
        return image

    def trainPredictor(self, predictor):
        '''Train the predictor with some random feature vectors.'''
        for i in range(30):
            predictor.add_image(self.createRandomImage(),
                                np.random.random())

        predictor.train()

class TestThumbnailSelector(unittest.TestCase):
    def setUp(self):
        self.model = model.load_model(os.path.join(os.path.dirname(__file__),
                                                   'simple_model.model'))

    def tearDown(self):
        pass

    @unittest.skip(("The testing model, simple_model.model, uses an outdated"
                    " version of the code and attempts to call accept_score"))
    def test_smoke_test(self):
        '''Looking for smoke when running the model on a small video.'''
        mov = cv2.VideoCapture(os.path.join(os.path.dirname(__file__),
                                            'test_videos',
                                            'swimmer.mp4'))

        thumb_data, _ = self.model.choose_thumbnails(mov, 5.0)

        self.assertGreater(len(thumb_data), 0)
        self.assertGreater(thumb_data[0][1], thumb_data[1][1])

if __name__ == '__main__':
    unittest.main()
