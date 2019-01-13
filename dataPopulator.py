import os
import json
import pymysql
import pymysql.cursors
import logging
import time

class DatabasePopulator(object):
    def __init__(self, cfg_filename):
        with open(cfg_filename, "r") as read_file:
            self.cfg = json.load(read_file)
            self.mysql_cfg = self.cfg.get("mysql_cfg")
            self.cols = self.cfg.get("table_cols")
            self.env_cfg = self.cfg.get("env_cfg")
            host = self.mysql_cfg.get('host')
            user = self.mysql_cfg.get('user')
            passwd = self.mysql_cfg.get('password')
            db = self.mysql_cfg.get('database')
        self.db = pymysql.connect(host=host, port=3306, user=user, passwd=passwd, db=db)
        self.cursor = self.db.cursor()
        logging.basicConfig(
            level= logging.INFO,
            filename=os.path.join(self.env_cfg.get('log_dir'), 'log.log'),
            format=self.env_cfg.get('log_format')
        )
        logging.Formatter.converter = time.gmtime
        self.log = logging.getLogger(self.__class__.__name__)

    def load_digit_struct(self, filepath):
        with open(filepath, "r") as read_file:
            data = json.load(read_file)
            return data

    def process_json_file(self, path, cuts):
        with open(path, "r") as read_file:
            data = json.load(read_file)
        images = list()
        for image in data:
            for i in range(0, len(image['boxes'])):
                filename = image['filename']
                boxNo = i
                label = image['boxes'][i]['label']
                width_val = image['boxes'][i]['width']
                top_val = image['boxes'][i]['top']
                left_val = image['boxes'][i]['left']
                height_val = image['boxes'][i]['height']
                stats = dict()
                for key in self.cols.keys():   
                    stats.update({key: eval(key)})
                images.append(stats)
        return images

    def add_data(self, images, table_name):
        """data is a list of dictionaries"""
        for image in images:
            cmd = ("INSERT IGNORE INTO " + table_name + ' (' +
                    ', '.join(self.cols.keys()) + ') VALUES ("' +
                    '", "'.join([str(image.get(k, 'NULL'))
                                for k in self.cols.keys()])
                    + '" )')
            try:
                self.cursor.execute(cmd)
            except Exception as ierr:
                self.log.error(ierr)
        self.db.commit()

    def create_table(self, tablename):
        cmd = "CREATE TABLE IF NOT EXISTS {} (".format(tablename)
        for col, typ in self.cols.items():
            cmd += " {} {} NOT NULL,".format(col, typ)
        cmd = cmd[:-1] + ' )'
        try:
            self.cursor.execute(cmd)
        except Exception as ierr:
            self.log.error(ierr)
        self.db.commit()

if __name__ == '__main__':
    populator = DatabasePopulator('./cfg.json')
    train_table = populator.env_cfg.get("train_table")
    train_path = populator.env_cfg.get("train_path")
    train_cuts = populator.env_cfg.get("train_cuts")
    test_table = populator.env_cfg.get("test_table")
    test_path = populator.env_cfg.get("test_path")
    test_cuts = populator.env_cfg.get("test_cuts")
    valid_table = populator.env_cfg.get("valid_table")
    valid_path = populator.env_cfg.get("valid_path")
    valid_cuts = populator.env_cfg.get("valid_cuts")
    populator.create_table("train")
    populator.create_table("test")
    populator.create_table("valid")
    images = populator.process_json_file(train_path, train_cuts)
    populator.add_data(images, train_table)
    images = populator.process_json_file(test_path, test_cuts)
    populator.add_data(images, test_table)
    images = populator.process_json_file(valid_path, valid_cuts)
    populator.add_data(images, valid_table)
