# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


"""
Standalone is designed for easy detect, track, and reid people in a single thread 
environment or in command line. 

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


from threading import Lock

_mt = None
_mt_lock = Lock()


def _get_mt():
    """Create the shared pipeline once, on its first facade call."""
    global _mt
    if _mt is None:
        with _mt_lock:
            if _mt is None:
                from .mt import MT
                _mt = MT()
    return _mt


def __getattr__(name):
    if name == 'MT':
        from .mt import MT
        globals()['MT'] = MT
        return MT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

def setConfigDir(config_dir=None, load_all=False):
    """See :func:`pyppbox.ppb.mt.MT.setConfigDir`"""
    _get_mt().setConfigDir(config_dir=config_dir, load_all=load_all)

def setMainModules(main_yaml=None, load_all=True):
    """See :func:`pyppbox.ppb.mt.MT.setMainModules`"""
    _get_mt().setMainModules(main_yaml=main_yaml, load_all=load_all)

def forceFullGTMode():
    """See :func:`pyppbox.ppb.mt.MT.forceFullGTMode`"""
    _get_mt().forceFullGTMode()

def setMainDetector(detector=""):
    """See :func:`pyppbox.ppb.mt.MT.setMainDetector`"""
    _get_mt().setMainDetector(detector=detector)

def getConfig():
    """See :func:`pyppbox.ppb.mt.MT.getConfig`"""
    return _get_mt().getConfig()

def getMainConfig(current=True):
    """See :func:`pyppbox.ppb.mt.MT.getMainConfig`"""
    return _get_mt().getMainConfig(current=current)

def detectPeople(
    img, 
    img_is_mat=False, 
    visual=False, 
    save=False, 
    save_file="", 
    min_width_filter=15, 
    alt_repspoint=False, 
    alt_repspoint_top=True
):
    """See :func:`pyppbox.ppb.mt.MT.detectPeople`"""
    return _get_mt().detectPeople(img,
                            img_is_mat=img_is_mat, 
                            visual=visual, 
                            save=save, 
                            save_file=save_file, 
                            min_width_filter=min_width_filter, 
                            alt_repspoint=alt_repspoint, 
                            alt_repspoint_top=alt_repspoint_top)

def setMainTracker(tracker=""):
    """See :func:`pyppbox.ppb.mt.MT.setMainTracker`"""
    _get_mt().setMainTracker(tracker=tracker)

def trackPeople(img, people, img_is_mat=False):
    """See :func:`pyppbox.ppb.mt.MT.trackPeople`"""
    return _get_mt().trackPeople(img, people, img_is_mat=img_is_mat)

def setMainReIDer(reider="", auto_load=True):
    """See :func:`pyppbox.ppb.mt.MT.setMainReIDer`"""
    _get_mt().setMainReIDer(reider=reider, auto_load=auto_load)

def reidPeople(img, people, deduplicate=True, img_is_mat=False):
    """See :func:`pyppbox.ppb.mt.MT.reidPeople`"""
    return _get_mt().reidPeople(img, people, deduplicate=deduplicate, img_is_mat=img_is_mat)

def trainReIDClassifier(reider="", train_data="", classifier_pkl=""):
    """See :func:`pyppbox.ppb.mt.MT.trainReIDClassifier`"""
    _get_mt().trainReIDClassifier(reider=reider, train_data=train_data, classifier_pkl=classifier_pkl)

def reset(frame_index=0):
    """See :meth:`pyppbox.ppb.mt.MT.reset`. Does not create an unused pipeline."""
    if _mt is not None:
        _mt.reset(frame_index=frame_index)

def close():
    """Close the shared pipeline; the next facade call creates a fresh instance.

    Does not initialize an unused pipeline. On cleanup failure, retain the closed
    instance so a later close() can retry. Call without concurrent facade activity.
    """
    global _mt
    with _mt_lock:
        if _mt is not None:
            _mt.close()
            _mt = None

__all__ = ['setConfigDir', 'setMainModules', 'forceFullGTMode', 'setMainDetector', 
           'getConfig', 'getMainConfig', 'detectPeople', 'setMainTracker', 'trackPeople', 
           'setMainReIDer', 'reidPeople', 'trainReIDClassifier', 'reset', 'close', 'MT']
