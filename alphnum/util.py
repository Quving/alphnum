#!/usr/bin/python

import os
from os import listdir
from os.path import isfile, join


class Util():
    @staticmethod
    def get_all_files_from(path, types=[".png", ".jpg"]):
        files = [os.path.join(path, f) for f in listdir(path) if
                 (isfile(join(path, f)) and (os.path.splitext(f)[1] in types))]
        return files
