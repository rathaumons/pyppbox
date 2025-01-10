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


"""
Standalone is designed for easy detect, track, and reid people in a single thread 
enviroment or in command line. 

For multithreading application, see :py:class:`pyppbox.ppb.mt.MT`.

Example:

>>> import cv2
>>> from pyppbox.standalone import (setMainDetector, detectPeople, 
>>>                                 setMainReIDer, reidPeople)
>>> from pyppbox.utils.visualizetools import visualizePeople
>>> 
>>> image = "data/gta.jpg"
>>> 
>>> # Detect people and save as visualized image detection_output.jpg using YOLO Classic
>>> setMainDetector("YOLO_Classic") # Set by name using internal configs
>>> detected_people, visual_image = detectPeople(
>>>     img=image,
>>>     visual=True,
>>>     save=True,
>>>     save_file="detection_output.jpg"
>>> )
>>> 
>>> # Re-identify the detected people using FaceNet
>>> setMainReIDer("FaceNet") # Set by name using internal configs
>>> reidentified_people, reid_count = reidPeople(img=image, people=detected_people)
>>> 
>>> # Visual people using pyppbox's visualizetools
>>> visualized_mat = visualizePeople(
>>>     img=image,
>>>     img_is_mat=False,
>>>     people=reidentified_people,
>>>     show_box=True,
>>>     show_skl=(True,True,5),
>>>     show_ids=(True,True,True),
>>>     show_repspoint=True,
>>> )
>>> cv2.imshow("Standalone", visualized_mat)
>>> cv2.waitKey(5000) # Show visualized_mat for 5 seconds
>>> 
>>> # Save the visualized_mat as reidentification_output.jpg
>>> cv2.imwrite("reidentification_output.jpg", visualized_mat)
>>> 

"""


from .mt import MT

_mt = MT()

def setConfigDir(config_dir=None, load_all=False):
    """See :func:`pyppbox.ppb.mt.MT.setConfigDir`"""
    _mt.setConfigDir(config_dir=config_dir, load_all=load_all)

def setMainModules(main_yaml=None, load_all=True):
    """See :func:`pyppbox.ppb.mt.MT.setMainModules`"""
    _mt.setMainModules(main_yaml=main_yaml, load_all=load_all)

def getConfig():
    """See :func:`pyppbox.ppb.mt.MT.getConfig`"""
    return _mt.getConfig()

def forceFullGTMode():
    """See :func:`pyppbox.ppb.mt.MT.forceFullGTMode`"""
    _mt.forceFullGTMode()

def setMainDetector(detector=""):
    """See :func:`pyppbox.ppb.mt.MT.setMainDetector`"""
    _mt.setMainDetector(detector=detector)

def detectPeople(img, 
                 img_is_mat=False, 
                 visual=False, 
                 save=False, 
                 save_file="", 
                 min_width_filter=35, 
                 alt_repspoint=False, 
                 alt_repspoint_top=True):
    """See :func:`pyppbox.ppb.mt.MT.detectPeople`"""
    return _mt.detectPeople(img, 
                            img_is_mat=img_is_mat, 
                            visual=visual, 
                            save=save, 
                            save_file=save_file, 
                            min_width_filter=min_width_filter, 
                            alt_repspoint=alt_repspoint, 
                            alt_repspoint_top=alt_repspoint_top)

def setMainTracker(tracker=""):
    """See :func:`pyppbox.ppb.mt.MT.setMainTracker`"""
    _mt.setMainTracker(tracker=tracker)

def trackPeople(img, people, img_is_mat=False):
    """See :func:`pyppbox.ppb.mt.MT.trackPeople`"""
    return _mt.trackPeople(img, people, img_is_mat=img_is_mat)

def setMainReIDer(reider="", auto_load=True):
    """See :func:`pyppbox.ppb.mt.MT.setMainReIDer`"""
    _mt.setMainReIDer(reider=reider, auto_load=auto_load)

def reidPeople(img, people, deduplicate=True, img_is_mat=False):
    """See :func:`pyppbox.ppb.mt.MT.reidPeople`"""
    return _mt.reidPeople(img, people, deduplicate=deduplicate, img_is_mat=img_is_mat)

def trainReIDClassifier(reider="", train_data="", classifier_pkl=""):
    """See :func:`pyppbox.ppb.mt.MT.trainReIDClassifier`"""
    _mt.trainReIDClassifier(reider=reider, train_data=train_data, classifier_pkl=classifier_pkl)

__all__ = ['setConfigDir', 'setMainModules', 'getConfig', 'forceFullGTMode', 
           'setMainDetector', 'detectPeople', 'setMainTracker', 'trackPeople', 
           'setMainReIDer', 'reidPeople', 'trainReIDClassifier', 'MT']
