# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from .mt import MT


__stdmt__ = MT()

def setConfigDir(config_dir=None, load_all=False):
    __stdmt__.setConfigDir(config_dir=config_dir, load_all=load_all)

def setMainModules(main_yaml=None, load_all=True):
    __stdmt__.setMainModules(main_yaml=main_yaml, load_all=load_all)

def getConfig():
    return __stdmt__.getConfig()

def forceFullGTMode():
    __stdmt__.forceFullGTMode()

def setMainDetector(detector=""):
    __stdmt__.setMainDetector(detector=detector)

def detectPeople(img, img_is_mat=False, visual=False, save=False, save_file=""):
    return __stdmt__.detectPeople(img, img_is_mat=img_is_mat, visual=visual, save=save, save_file=save_file)

def setMainTracker(tracker=""):
    __stdmt__.setMainTracker(tracker=tracker)

def trackPeople(img, people, img_is_mat=False):
    return __stdmt__.trackPeople(img, people, img_is_mat=img_is_mat)

def setMainReIDer(reider="", auto_load=True):
    __stdmt__.setMainReIDer(reider=reider, auto_load=auto_load)

def reidPeople(img, people, deduplicate=True, img_is_mat=False):
    return __stdmt__.reidPeople(img, people, deduplicate=deduplicate, img_is_mat=img_is_mat)

def trainReIDClassifier(reider="Default", train_data="", classifier_pkl=""):
    __stdmt__.trainReIDClassifier(reider=reider, train_data=train_data, classifier_pkl=classifier_pkl)

