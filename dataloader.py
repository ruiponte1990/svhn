import os
import json
import numpy as np
import pymysql
import pymysql.cursors
import logging
import pandas as pd
import tensorflow as tf
from skimage.color import rgb2gray
from sklearn.preprocessing import LabelBinarizer
import cv2
import time

class Dataloader(object):
    def __init__(self, cfg):
        with open(cfg, 'r') as cfg:
            self.cfg = json.load(cfg)
        self.sql_cfg = self.cfg.get("mysql_config")
        self.env_cfg = self.cfg.get("env_cfg")
        self.cols = self.cfg.get("table_cols")
        self.db = pymysql.connect(host='localhost', port=3306, user='root', passwd='password', db='svhn')
        self.cursor = self.db.cursor()
        logging.basicConfig(
            level= logging.INFO,
            filename=os.path.join(self.env_cfg.get('log_dir'), 'log.log'),
            format=self.env_cfg.get('log_format')
        )
        logging.Formatter.converter = time.gmtime
        self.log = logging.getLogger(self.__class__.__name__)

    def load_meta_data(self, filename, tablename):
        cmd = "SELECT * from {} WHERE filename = \'{}\';".format(tablename, filename)
        self.cursor.execute(cmd)
        results = self.cursor.fetchall()
        results = np.array(results)
        df = pd.DataFrame(results, columns=self.cols.keys())
        return df

    def load_raw_pixels(self, filename, tablename):
        path = '../data/'+ tablename +'/'
        filename = path + filename
        return cv2.imread(filename)

    def load_data(self, filename, tablename):
        imageData = self.load_raw_pixels(filename, tablename)
        df = self.load_meta_data(filename, tablename)
        return {"meta": df, "raw": imageData}
    
    def grab_boxes(self, df):
        meta = df["meta"]
        raw = df["raw"]
        boxes = []
        for index, row in meta.iterrows():
            try:
                box = self.grab_box(int(float(row['width_val'])), int(float(row['top_val'])), int(float(row['left_val'])), int(float(row['height_val'])),raw, row['filename'])
                max = box.max()
                box = (box * (255/max))
                label = int(float(row['label']))
                binary = LabelBinarizer()
                binary.fit([0,1,2,3,4,5,6,7,8,9])
                oneHot = binary.transform([label])
                tup = (box, oneHot)
                boxes.append(tup)
            except Exception:
                print('Grab Box Error: ', row['filename'])
                continue
        return boxes

    def grab_box(self, width_val, top_val, left_val, height_val, raw, filename):
        box = raw[top_val:(height_val+top_val), left_val:(width_val+left_val)]
        try:
            box = cv2.resize(box,(128,128))
            box = rgb2gray(box)
        except Exception:
            print('Resize Error: ', filename)
            return
        return box