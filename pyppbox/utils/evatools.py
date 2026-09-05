# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import numpy as np
from scipy.optimize import linear_sum_assignment

from .gttools import GTInterpreter, convertStringToNPL
from .commontools import joinFPathFull, getAbsPathFDS, isExist, getAncestorDir
from .logtools import add_info_log, add_warning_log, add_error_log


class MyEVA(object):

    """A class used to generate the evaluation of the supported datasets.

    Attributes
    ----------
    reid_count : int
        Total number of ReID count.
    diff_count : int
        Total number of wrong IDs.
    missed_detect : int
        Total number of missed detection(s).
    fault_detect : int
        Total number of fault detection(s).
    current_frame : int
        Current frame index.
    frame_to_check : int
        Frame index which is used for the validation.
    score : float
        Unclamped score from the accumulated counts and the full GT row count.
    id_mode : str
        Defaults to ``"deepid"``.
        Indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
    gt_interpreter : pyppbox.utils.gttools.GTInterpreter
        Set automatically.
        GT (Ground-truth) loader, :class:`~pyppbox.utils.gttools.GTInterpreter` object.
    gt_file_name : str
        Set automatically.
        GT (Ground-truth) file name.
    gt_file : str
        Set automatically.
        GT (Ground-truth) file path.

    Notes
    -----
    Use a new instance for each evaluation run. Selecting another GT file does not
    reset accumulated counts or the frame cursor. ``validate()`` updates identity,
    missed-detection, and false-detection counts; ``reid_count`` is supplied separately
    through ``setReIDcount()``.
    """

    def __init__(self):
        self.reid_count = 0
        self.diff_count = 0
        self.missed_detect = 0
        self.fault_detect = 0
        self.current_frame = 0
        self.frame_to_check = 0
        self.id_mode = "deepid"
        self.no_gt = False
        self.gt_file = ""
        self.gt_interpreter = GTInterpreter()

    def setGTByGTMap(self, gt_map_txt, input_video, id_mode="deepid"):
        """Set a GT (Ground-truth) text file by give the GT mapping file ``gt_map_txt`` and
        ``input_video`` of a supported dataset.

        Parameters
        ----------
        gt_map_txt : str
            A file path of a Video:GT mapping text file for GT (Ground-truth).
        input_video : str
            A video file path.
        id_mode : str
            Defaults to ``"deepid"``.
            An indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
        """
        self.no_gt = False
        if str(id_mode).lower() in ("deepid", "faceid"):
            self.id_mode = str(id_mode).lower()
        if id_mode != "deepid":
            if str(id_mode).lower() in ("deepid", "faceid"):
                self.id_mode = str(id_mode).lower()
                add_info_log(f"-------EVA : Set id_mode='{self.id_mode}'")
            else :
                add_warning_log(f"-------EVA : id_mode='{id_mode}' is not recognized.")
                add_warning_log(f"-------EVA : Override id_mode='{self.id_mode}'.")
        self.gt_interpreter = GTInterpreter()
        self.gt_interpreter.gtIO.loadInputGTMap(gt_map_txt)
        self.gt_file_name = self.gt_interpreter.gtIO.getGTFileName(input_video)
        self.gt_file = ""
        if self.gt_file_name != "":
            self.gt_file = joinFPathFull(getAncestorDir(gt_map_txt), self.gt_file_name)
            self.gt_interpreter.setGT(self.gt_file)
        else:
            msg = f"MyEVA : setGTByGTMap() -> There is no GT file for the input '{input_video}'"
            # add_error_log(msg)
            # raise ValueError(msg)
            add_warning_log(msg)
            self.no_gt = True

    
    def setGTByKnownGTFile(self, gt_file, id_mode="deepid"):
        """Set a GT (Ground-truth) text file and :obj:`id_mode` which is used to compare.

        Parameters
        ----------
        gt_file : str
            A file path of a GT (Ground-truth).
        id_mode : str
            Defaults to ``"deepid"``.
            An indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
        """
        self.no_gt = False
        if str(id_mode).lower() in ("deepid", "faceid"):
            self.id_mode = str(id_mode).lower()
        if id_mode != "deepid":
            if str(id_mode).lower() in ("deepid", "faceid"):
                self.id_mode = str(id_mode).lower()
                add_info_log(f"-------EVA : Set id_mode='{self.id_mode}'")
            else :
                add_warning_log(f"-------EVA : id_mode='{id_mode}' is not recognized.")
                add_warning_log(f"-------EVA : Override id_mode='{self.id_mode}'.")
        self.gt_interpreter = GTInterpreter()
        self.gt_file = gt_file
        if isExist(gt_file):
            self.gt_interpreter.setGT(getAbsPathFDS(self.gt_file))
            add_info_log(f"-------EVA : Custom gt_file='{gt_file}'")
        else:
            msg = f"MyEVA : setGTByKnownGTFile() -> The input gt_file='{gt_file}' does not exist"
            # add_error_log(msg)
            # raise ValueError(msg)
            add_warning_log(msg)
            self.no_gt = True

    def __checkInReID__(self):
        self.reid_count += 1

    def setReIDcount(self, total_count):
        """Set the total :attr:`reid_count` according to ``total_count``.

        Parameters
        ----------
        total_count : int
            Total number of ReID count.
        """
        self.reid_count = total_count

    def __compareIDList2GT__(self, id_list_gt, id_list_dt):
        diff_count = 0
        missed_detect = 0
        fault_detect = 0
        tmp = len(id_list_gt) - len(id_list_dt)
        i = 0
        if tmp >= 0:
            missed_detect = tmp
            for i in range(0, len(id_list_dt)):
                if '%' in id_list_dt[i]: id_list_dt[i] = id_list_dt[i][:-4]
                if id_list_gt[i] != id_list_dt[i]:
                    msg = (f"-------EVA : ------------------------------>   "
                           f"{id_list_gt[i]}\t  --vs--    {id_list_dt[i]}\t@{self.current_frame}")
                    add_info_log(msg)
                    diff_count += 1
        else:
            fault_detect = len(id_list_dt) - len(id_list_gt)
            for i in range(0, len(id_list_gt)):
                if '%' in id_list_dt[i]: id_list_dt[i] = id_list_dt[i][:-4]
                if id_list_gt[i] != id_list_dt[i]:
                    msg = (f"-------EVA : ------------------------------>   "
                           f"{id_list_gt[i]}\t  --vs--    {id_list_dt[i]}\t@{self.current_frame}")
                    add_info_log(msg)
                    diff_count += 1
        return diff_count, missed_detect, fault_detect

    def __compareDeepID__(self, gt_frame, people_dt):
        return _compare_detections(
            [convertStringToNPL(row[4]) for row in gt_frame], [row[2] for row in gt_frame],
            [person.box_xyxy for person in people_dt], [person.deepid for person in people_dt], 16)

    def __compareFaceID__(self, gt_frame, people_dt):
        return _compare_detections(
            [convertStringToNPL(row[4]) for row in gt_frame], [row[2] for row in gt_frame],
            [person.box_xyxy for person in people_dt], [person.faceid for person in people_dt], 16)

    def validate(self, people, frame_id=-1):
        """Accumulate one frame's identity and detection errors using one-to-one box matching.

        Parameters
        ----------
        people : list[pyppbox.utils.persontools.Person]
            Predicted people. Use an empty list to count missing detections in a GT frame.
        frame_id : int
            Defaults to ``-1``. A nonnegative value selects an explicit GT frame for
            this call. Otherwise use ``current_frame``.

        Notes
        -----
        Returns None. Matching allows a maximum absolute difference of 16 pixels in
        any corresponding xyxy coordinate. Identity comparison is case-insensitive.
        Unmatched references and predictions are counted independently. Each call
        increments ``current_frame`` by one, even when an explicit frame ID is supplied
        or GT is unavailable; it does not set the cursor to ``frame_id + 1``.
        Repeated calls for the same frame accumulate errors again.
        """
        if frame_id >= 0: self.frame_to_check = frame_id
        else : self.frame_to_check = self.current_frame
        if self.gt_file != "" and not self.no_gt:
            diff_c = 0
            missed_d = 0
            fault_d = 0
            gt_frame = self.gt_interpreter.findGTFrame(self.frame_to_check)
            if self.id_mode == "deepid":
                diff_c, missed_d, fault_d = self.__compareDeepID__(gt_frame=gt_frame, people_dt=people)
            elif self.id_mode == "faceid":
                diff_c, missed_d, fault_d = self.__compareFaceID__(gt_frame=gt_frame, people_dt=people)
            self.diff_count = self.diff_count + diff_c
            self.missed_detect = self.missed_detect + missed_d
            self.fault_detect = self.fault_detect + fault_d
        self.current_frame += 1

    def getSummary(self, print_summary=True):
        """Generate a summary of the evaluation.

        Parameters
        ----------
        print_summary : bool
            Defaults to ``True``.
            An indication of whether to print a summary text in the terminal.

        Returns
        -------
        diff_count : int
            Total number of wrong ID(s).
        missed_detect : int
            Total number of missed detection(s).
        fault_detect : int
            Total number of fault detection(s).
        reid_count : int
            Total number of ReID count.
        gt_interpreter.total_detections : int
            Total number of detection(s) or all ID(s) in all frame(s) in the GT (Ground-truth).
        score : float
            (Total ID - Wrong ID - Missed Detection) / Total ID; 0.0 for empty GT.
            False detections are reported separately, as in earlier V3 releases.

        Notes
        -----
        The denominator counts the entire GT file, including frames not yet validated.
        The method does not reset counts. Repeated-frame evaluation can produce a
        negative score because the formula is not clamped.
        """
        if self.no_gt or self.gt_interpreter.total_detections == 0:
            self.score = 0.0
        else:
            self.score = float((self.gt_interpreter.total_detections - self.diff_count - 
                                self.missed_detect) / self.gt_interpreter.total_detections)
        if print_summary:
            msg = (f"\n#####################################################################\n\n"
                   f"  Summary: \n\n"
                   f"  -----------------------------------------------------------------  \n"
                   f"                  ReID count  =  {self.reid_count}\n"
                   f"      Missed detection count  =  {self.missed_detect}\n"
                   f"       Fault detection count  =  {self.fault_detect}\n"
                   f"              Wrong ID count  =  {self.diff_count} / {self.gt_interpreter.total_detections}\n"
                   f"  -----------------------------------------------------------------  \n\n"
                   f"               * Final score  =  {self.score}\n\n"
                   f"     [(Total ID) - (Wrong ID) - (Missed Detection)] / (Total ID)     \n\n"
                   f"#####################################################################\n")
            add_info_log(msg, add_new_line=True)
        return (self.diff_count, self.missed_detect, self.fault_detect, self.reid_count, 
                self.gt_interpreter.total_detections, self.score)


class NothingDetecter(object):
    """
    A class acted as a detector which does not perform any detection. 
    """
    def __init__(self):
        pass
    def detectFrame(self):
        """Return a new empty detection list without inspecting an image.
        """
        return []


class NothingTracker(object):
    """Pass detections through without estimating motion or assigning new IDs.

    The input list and person objects are retained by reference. For a matching
    deep identity in the previous call, ``update()`` carries its ``misc`` data forward.
    """
    def __init__(self):
        self.previous_list = []
        self.current_list = []
    def _getIndex(self, deepid):
        pindex = -1
        for i in range(0, len(self.previous_list)):
            if deepid == self.previous_list[i].deepid:
                pindex = i
                break
        return pindex
    def update(self, pp, img=None):
        """Pass through this frame's people and carry matching deep-ID metadata forward.

        Parameters
        ----------
        pp : list[pyppbox.utils.persontools.Person]
            Current people. Objects and the list are not copied.
        img : object
            Defaults to ``None``. Ignored.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            The supplied list, with ``misc`` possibly updated from the previous call.
        """
        self.previous_list = self.current_list
        self.current_list = pp
        for i in range (0, len(self.current_list)):
            if len(self.previous_list) > 0:
                pindex = self._getIndex(self.current_list[i].deepid)
                if pindex >= 0:
                    self.current_list[i].misc = self.previous_list[pindex].misc

        return self.current_list


class NothingReider(object):
    """
    A class acted as a reider which does not perform any re-identifying. 
    """
    def __init__(self):
        pass
    def recognize(self, res):
        """Return the supplied identity value unchanged.

        Parameters
        ----------
        res : object
            Identity value to pass through.

        Returns
        -------
        object
            The same value that was supplied.
        """
        return res


class TKOReider(object):

    """Generate fallback identity strings without examining an image.

    Random mode uses a prefix of an uppercase UUID. Static mode consumes a stored
    sequence and then returns ``EoR<index>`` labels when it is exhausted. These are
    fallback labels, not recognized identities, and random prefixes can collide.
    """

    def __init__(self, static=False, static_ids=None, string_length=5):
        """Initialize random or sequential fallback identities.

        Parameters
        ----------
        static : bool
            Defaults to ``False``. Consume stored IDs sequentially when True.
        static_ids : list[str] or None
            Defaults to ``None``. Optional explicit sequence, copied into the instance.
            In static mode, None or an empty sequence creates the built-in label list
            followed by 65,536 random prefixes.
        string_length : int
            Defaults to ``5``. Prefix length used for UUID-based labels; UUID hex strings
            contain at most 32 characters.
        """
        self.is_static = static
        self.string_length = string_length
        self.static_index = 0
        self.setStaticIDs(static_ids)

    def recognize(self, _):
        """Generate a fallback identity, ignoring the supplied input.

        Parameters
        ----------
        _ : object
            Ignored image or identity value.

        Returns
        -------
        str
            Next static label or a random UUID prefix, according to the selected mode.
        """
        res = ""
        if self.is_static:
            res = self.generateStaticID()
        else:
            res = self.generateID(self.string_length)
        return res

    def generateID(self, string_length):
        """Return an uppercase hexadecimal UUID prefix.

        Parameters
        ----------
        string_length : int
            Slice length for the 32-character UUID hex string.

        Returns
        -------
        str
            The UUID prefix. Short prefixes are not guaranteed unique.
        """
        import uuid
        random = str(uuid.uuid4()) # Convert UUID format to a Python string.
        random = random.upper() # Make all characters uppercase.
        random = random.replace("-","") # Remove the UUID '-'.
        res = random[0:string_length]
        return res

    def generateStaticID(self):
        """Return the next stored ID and increment the sequence index.

        Returns
        -------
        str
            Stored ID, or ``EoR<index>`` once the explicit sequence has been exhausted.
        """
        try:
            res = self.static_ids[self.static_index]
        except IndexError:
            add_warning_log("-TKOReider : Random ID is out of range")
            res = "EoR" + str(self.static_index)
        self.static_index += 1
        return res

    def setStaticIDs(self, static_ids, plus_random=65536):
        """Replace the static sequence and reset its index to zero.

        Parameters
        ----------
        static_ids : list[str] or None
            Sequence to copy. The caller's list is not modified.
        plus_random : int
            Defaults to ``65536``. Number of UUID prefixes appended to the built-in
            defaults only when static mode receives None or an empty sequence. An
            explicit nonempty sequence is never extended.

        Notes
        -----
        Returns None. In random mode, the copied sequence is stored but not consumed
        by ``recognize()``.
        """
        self.static_ids = [] if static_ids is None else list(static_ids)
        self.static_index = 0
        self.static_ids_len = len(self.static_ids)
        if self.is_static:
            if self.static_ids_len <= 0:
                default_ids = ["Lester", "Michael", "Franklin", "Trevor", "Amanda", "MCU-Vision", "MCU-Thor",
                               "MCU-Hulk", "MCU-Loki", "MCU-Thanos", "DC-Batman", "DC-Superman", "DC-Aquaman",
                               "DC-Shazam", "DC-Cyborg"]
                self.static_ids.extend(default_ids)
                for _ in range(plus_random):
                    self.static_ids.append(self.generateID(self.string_length))
        self.static_ids_len = len(self.static_ids)




###############################################################################################


def _compare_detections(ref_boxes, ref_ids, predicted_boxes, predicted_ids, max_spread):
    """Match geometry one-to-one before comparing identities; never reuse a reference."""
    nr, npred = len(ref_boxes), len(predicted_boxes)
    if nr == 0 or npred == 0:
        return 0, nr, npred
    reference = np.asarray(ref_boxes, dtype=float).reshape(nr, 4)
    predicted = np.asarray(predicted_boxes, dtype=float).reshape(npred, 4)
    distances = np.max(np.abs(reference[:, None, :] - predicted[None, :, :]), axis=2)
    # An invalid edge costs more than all possible valid edges combined, so the
    # assignment maximizes valid matches first and minimizes their distance second.
    invalid_cost = (min(nr, npred) + 1) * (float(max_spread) + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= max_spread, distances, invalid_cost))
    pairs = [(r, c) for r, c in zip(rows, cols) if distances[r, c] <= max_spread]
    wrong = 0
    for r, c in pairs:
        identity = str(predicted_ids[c])
        if '%' in identity:
            identity = identity[:-4]  # Legacy identity strings can contain a confidence suffix.
        wrong += str(ref_ids[r]).lower() != identity.lower()
    return wrong, nr - len(pairs), npred - len(pairs)


def findPersonIndexGTFrame(gt_frame, box_xyxy, box_xyxy_index=4, max_spread_limit=16):
    """
    :meta private:
    """
    box_list = box_xyxy.tolist()
    min_box_spread = 8192
    index = -1
    i = 0
    for p in gt_frame:
        pbbox_list = convertStringToNPL(p[box_xyxy_index]).tolist()
        max_ss = max([abs(box_list[j] - pbbox_list[j]) for j in range(0, 4)])
        if max_ss < min_box_spread:
            min_box_spread = max_ss
            index = i
        i += 1
    if min_box_spread > max_spread_limit: index = -1
    return index

def compareRes2Ref(res_txt, ref_txt, res_box_xyxy_index=5, ref_box_xyxy_index=4, 
                   res_compare_index=2, ref_compare_index=2, box_max_spread=5):
    """Compare the result text file generated by :class:`~pyppbox.utils.restools.ResIO` to any reference
    or GT (Ground-truth) text file across all frame IDs in either file, including
    prediction-only frames. This is used for comparing the strings of
    ``deepid`` or ``faceid`` in result to the reference.

    Parameters
    ----------
    res_txt : str
        A path of the result text file.
    ref_txt : str
        A path of the reference text file.
    res_box_xyxy_index : int
        Defaults to ``5``.
        Index of bounding box :code:`[X1, Y1, X2, Y2]` in the result text file.
    ref_box_xyxy_index : int
        Defaults to ``4``.
        Index of bounding box :code:`[X1, Y1, X2, Y2]` in the reference text file.
    res_compare_index : int
        Defaults to ``2``.
        Index of what to compare in the result text file.
    ref_compare_index : int
        Defaults to ``2``.
        Index of what to compare in the reference text file.
    box_max_spread : int
        Defaults to ``5``.
        Max spread or max margin used to decide whether 2 bounding boxes are the same 
        by comparing the differences between the elements in the result's bounding 
        box and the coressponding elements in the reference's bounding box.
    
    Returns
    -------
    int
        Total number of wrong ID count.
    int
        Total number of missed detection count.
    int
        Total number of fault detection count.
    int
        Total number of all detection or ID count in all frame(s) in the reference text file.
    float
        (Total ID - Wrong ID - Missed detection) / Total ID; 0.0 for empty GT.
        False detections are reported separately and are not subtracted from this score.
    """

    diff_count = 0
    missed_detect = 0
    fault_detect = 0
    score = 0
    total_detections = 0
    
    if isExist(str(ref_txt)):
        ref_interpreter = GTInterpreter()
        ref_interpreter.setGT(getAbsPathFDS(ref_txt))
        
        if isExist(str(res_txt)):
            res_interpreter = GTInterpreter()
            res_interpreter.setGT(getAbsPathFDS(res_txt))
            # Include sparse/nonzero frame IDs and prediction-only frames exactly once.
            frames = sorted(set(ref_interpreter.gt_frames_list) | set(res_interpreter.gt_frames_list))
            for frame in frames:
                ref_frame = ref_interpreter.findGTFrame(frame)
                res_frame = res_interpreter.findGTFrame(frame)
                wrong, missed, false = _compare_detections(
                    [convertStringToNPL(row[ref_box_xyxy_index]) for row in ref_frame],
                    [row[ref_compare_index] for row in ref_frame],
                    [convertStringToNPL(row[res_box_xyxy_index]) for row in res_frame],
                    [row[res_compare_index] for row in res_frame], box_max_spread)
                diff_count += wrong
                missed_detect += missed
                fault_detect += false

            total_detections = ref_interpreter.total_detections
            score = (float((total_detections - diff_count - missed_detect) / total_detections)
                     if total_detections else 0.0)

            msg = (f"\n#####################################################################\n\n"
                   f"  Summary: \n\n"
                   f"  -----------------------------------------------------------------  \n"
                   f"      Missed detection count  =  {missed_detect}\n"
                   f"       Fault detection count  =  {fault_detect}\n"
                   f"              Wrong ID count  =  {diff_count} / {total_detections}\n"
                   f"  -----------------------------------------------------------------  \n\n"
                   f"               * Final score  =  {score}\n\n"
                   f"     [(Total ID) - (Wrong ID) - (Missed Detection)] / (Total ID)     \n\n"
                   f"#####################################################################\n")

            add_info_log(msg, add_new_line=True)

        else:
            msg = "compareRes2Ref() -> Input 'res_txt' does not exist. "
            add_error_log(msg)
            raise ValueError(msg)
    else:
        msg = "compareRes2Ref() -> Input 'ref_txt' does not exist. "
        add_error_log(msg)
        raise ValueError(msg)
    
    return diff_count, missed_detect, fault_detect, total_detections, score
