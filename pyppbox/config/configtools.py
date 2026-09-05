# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2025 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU Affero General Public License as          #
#   published by the Free Software Foundation, either version 3 of the      #
#   License, or (at your option) any later version.                         #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU Affero General Public License for more details.                     #
#                                                                           #
#   You should have received a copy of the GNU Affero General Public License#
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import json
import os
import stat
import tempfile
import yaml

from .unifiedstrings import UnifiedStrings
from pyppbox.utils.commontools import joinFPathFull, getAbsPathFDS, getGlobalRootDir, isExist
from pyppbox.utils.logtools import add_warning_log, add_error_log


internal_root_dir = getGlobalRootDir()
internal_config_dir = joinFPathFull(internal_root_dir, 'config')
internal_cfg_dir = joinFPathFull(internal_config_dir, 'cfg')

class PYPPBOXStructure(object):

    """
    A class used to organize the structure of pyppbox.

    Attributes
    ----------
    cfg_dir : str
        Defaults to ``'{pyppbox root}/config/cfg'``.
        Path of config directory where stores main.yaml, detectors.yaml, 
        trackers.yaml, reiders.yaml.
    internal_root_dir : str
        Set automatically.
        Path of pyppbox's root directory.
    gui_root : str
        Set automatically.
        Path of GUI's root directory.
    gui_tmp_dir : str
        Set automatically.
        Path of GUI's tmp directory.
    data_dir : str
        Set automatically.
        Internal path of pyppbox's data directory.
    dataset_dir : str
        Set automatically.
        Path of a supported dataset like 
        :code:`{pyppbox root}/data/datasets/GTA_V_DATASET`, etc.
    gt_dir : str
        Set automatically.
        Path of a dataset's GT (Ground-truth) directory where stores all 
        the ground-truth text files and the mapping text file.
    main_yaml : str
        Set automatically.
        Path of main.yaml.
    detectors_yaml : str
        Set automatically.
        Path of detectors.yaml.
    trackers_yaml : str
        Set automatically.
        Path of trackers.yaml.
    reiders_yaml : str
        Set automatically.
        Path of reiders.yaml.
    unified_strings : pyppbox.config.unifiedstrings.UnifiedStrings
        Set automatically.
        A :class:`~pyppbox.config.unifiedstrings.UnifiedStrings` object used to store unified strings.
    """

    def __init__(self, cfg_dir=internal_cfg_dir):
        """Initialize paths according to ``cfg_dir`` and automatically call
        :meth:`setYAMLPath()`.

        Parameters
        ----------
        cfg_dir : str
            Defaults to ``'{pyppbox root}/config/cfg'``.
            A path of the config directory where stores main.yaml, detectors.yaml, 
            trackers.yaml, and reiders.yaml.
        """
        if cfg_dir != internal_cfg_dir:
            if not isExist(cfg_dir):
                add_warning_log(f"PYPPBOXStructure : __init__() -> cfg_dir='{cfg_dir}' does not exist.")
                add_warning_log("PYPPBOXStructure : __init__() -> Switched to internal cfg directory !")
                cfg_dir = internal_cfg_dir
        self.cfg_dir = cfg_dir
        self.internal_root_dir = internal_root_dir
        self.unified_strings = UnifiedStrings()
        self.setDIR()
        self.setYAMLPath()

    def setDIR(self):
        """Automatically config or set all necessary directories of the base structure 
        of pyppbox such as :attr:`gui_root`, :attr:`gui_tmp_dir`, :attr:`data_dir`, 
        :attr:`dataset_dir`, and :attr:`gt_dir`.
        """
        # GUI 
        self.gui_root = joinFPathFull(self.internal_root_dir, 'gui')
        self.gui_tmp_dir = joinFPathFull(self.gui_root, 'tmp')
        # Dataset
        self.data_dir = joinFPathFull(self.internal_root_dir, 'data')
        self.dataset_dir = joinFPathFull(self.data_dir, 'datasets/GTA_V_DATASET')
        self.gt_dir = joinFPathFull(self.dataset_dir, 'ground_truth')

    def setYAMLPath(self):
        """
        Set the paths of the necessary YAML files such as :attr:`main_yaml`, 
        :attr:`detectors_yaml`, :attr:`trackers_yaml`, and :attr:`reiders_yaml`.
        """
        self.main_yaml = joinFPathFull(self.cfg_dir, "main.yaml")
        self.detectors_yaml = joinFPathFull(self.cfg_dir, "detectors.yaml")
        self.trackers_yaml = joinFPathFull(self.cfg_dir, "trackers.yaml")
        self.reiders_yaml = joinFPathFull(self.cfg_dir, "reiders.yaml")
    
    def setCustomCFG(self, cfg_dir):
        """Set a path of a custom config directory where stores main.yaml, 
        detectors.yaml, trackers.yaml, and reiders.yaml.

        Parameters
        ----------
        cfg_dir : str
            A path of the config directory.
        """
        if not isExist(cfg_dir):
            add_warning_log(f"PYPPBOXStructure : setCustomCFG() -> cfg_dir='{cfg_dir}' does not exist.")
            add_warning_log("PYPPBOXStructure : setCustomCFG() -> Switched to internal cfg directory!")
            cfg_dir = internal_cfg_dir
        self.cfg_dir = cfg_dir
        self.setYAMLPath()


#########################################################################################


def _normalize_documents(value):
    """Normalize a mapping or list of mappings without changing config values."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(doc, dict) for doc in value):
        return value
    raise ValueError("Expected a configuration mapping or a list of mappings.")


def _single_document(documents):
    if len(documents) > 1:
        raise ValueError("Expected one configuration document; received multiple documents.")
    return documents[0] if documents else {}


def _parse_documents(raw_string):
    documents = []
    # Consume the generator here so errors in later YAML documents are caught too.
    for value in yaml.safe_load_all(raw_string):
        if value is not None:  # Empty YAML documents are harmless separators.
            documents.extend(_normalize_documents(value))
    return documents


def isDictString(input_string):
    """Check whether a string parses into at least one configuration mapping.

    Parameters
    ----------
    input_string : object
        Candidate inline YAML/JSON value. Non-strings return False.

    Returns
    -------
    bool
        True when at least one mapping is parsed, including an empty mapping
        (``{}``), list-wrapped mappings, and multiple YAML documents. Invalid
        input or a stream with no mappings returns False.
    """
    if not isinstance(input_string, str):
        return False
    try:
        return bool(_parse_documents(input_string))
    except (ValueError, yaml.YAMLError):
        return False


def isConfigInput(value):
    """Classify a value as custom configuration input rather than a short module name.

    Parameters
    ----------
    value : object
        Candidate module selector or configuration value.

    Returns
    -------
    bool
        True for mappings, lists, path-like objects, and strings resembling inline
        configs or config filenames. This is a routing heuristic: it does not
        validate content or check that a file exists.
    """
    if isinstance(value, (dict, list, os.PathLike)):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (stripped.startswith(('{', '[', '---')) or ':' in stripped
            or '\n' in stripped or _is_config_path(stripped))


def _is_config_path(value):
    # Inline config values can themselves end in '.yaml' or '.json'.
    if value.lstrip().startswith(('{', '[', '---')) or '\n' in value or ': ' in value:
        return False
    return os.path.splitext(value)[1].lower() in ('.yaml', '.yml', '.json')


def getCFGDict(input):
    """Load one configuration mapping.

    Parameters
    ----------
    input : str or dict or list or object
        A mapping, mapping list, raw YAML/JSON string, or YAML/JSON file path. Path-like objects are accepted.

    Returns
    -------
    dict
        One mapping, or an empty dictionary for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read. Multiple mappings are rejected.

    Notes
    -----
    Ready mappings/lists are returned without deep copying. Relative filenames
    resolve from the working directory. These helpers parse configuration values;
    they do not select modules or resolve model paths within the mappings.
    """
    return _single_document(getCFGDictList(input))


def getCFGDictList(input):
    """Load a list of configuration mappings.

    Parameters
    ----------
    input : str or dict or list or object
        A mapping, mapping list, raw YAML/JSON string, or YAML/JSON file path. Path-like objects are accepted.

    Returns
    -------
    list[dict]
        Mappings in input order, or an empty list for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read.

    Notes
    -----
    Ready mappings/lists are returned without deep copying. Relative filenames
    resolve from the working directory. These helpers parse configuration values;
    they do not select modules or resolve model paths within the mappings.
    """
    if isinstance(input, os.PathLike):
        return loadDocumentList(getAbsPathFDS(input))
    if isinstance(input, str):
        if _is_config_path(input):
            return loadDocumentList(getAbsPathFDS(input))
        return loadRawYAMLStringMT(input)
    return _normalize_documents(input)


def loadDocument(yaml_json):
    """Load one configuration mapping.

    Parameters
    ----------
    yaml_json : str or object
        A file path or path-like object. A .json suffix selects the JSON reader; other suffixes use YAML.

    Returns
    -------
    dict
        One mapping, or an empty dictionary for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read. Multiple mappings are rejected.
    """
    return _single_document(loadDocumentList(yaml_json))


def loadDocumentList(yaml_json):
    """Load a list of configuration mappings.

    Parameters
    ----------
    yaml_json : str or object
        A file path or path-like object. A .json suffix selects the JSON reader; other suffixes use YAML.

    Returns
    -------
    list[dict]
        Mappings in input order, or an empty list for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read.
    """
    try:
        with open(yaml_json, 'r', encoding='utf-8-sig') as cfg:
            if os.path.splitext(os.fspath(yaml_json))[1].lower() == '.json':
                return _normalize_documents(json.load(cfg))
            return _parse_documents(cfg)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as e:
        msg = f"loadDocumentList() -> '{yaml_json}': {e}"
        add_error_log(msg)
        raise ValueError(msg) from e


def loadRawYAMLString(raw_string):
    """Load one configuration mapping.

    Parameters
    ----------
    raw_string : str
        Inline YAML/JSON mapping text or a legacy list containing one mapping.

    Returns
    -------
    dict
        One mapping, or an empty dictionary for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read. Multiple mappings are rejected.
    """
    return _single_document(loadRawYAMLStringMT(raw_string))


def loadRawYAMLStringMT(raw_string):
    """Load a list of configuration mappings.

    Parameters
    ----------
    raw_string : str
        Inline YAML/JSON mappings, mapping lists, or a multi-document YAML stream.

    Returns
    -------
    list[dict]
        Mappings in input order, or an empty list for empty input.
        Empty YAML documents are skipped. Only mappings and lists of mappings
        are valid document values.

    Raises
    ------
    ValueError
        If input is malformed, contains a non-mapping document, or a requested
        file cannot be read.
    """
    try:
        return _parse_documents(raw_string)
    except (ValueError, yaml.YAMLError) as e:
        msg = f'loadRawYAMLStringMT() -> {e}'
        add_error_log(msg)
        raise ValueError(msg) from e


def _dump_documents(output_file, documents, header, single):
    temporary = None
    try:
        documents = _normalize_documents(documents)
        if os.path.splitext(os.fspath(output_file))[1].lower() == '.json':
            value = _single_document(documents) if single else documents
            content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        else:
            content = yaml.safe_dump_all(documents, sort_keys=False, allow_unicode=True)
            if header:
                content = header + ('' if header.endswith('\n') else '\n') + content
        # Serialize before opening anything, then replace only a fully written file.
        destination = os.path.realpath(os.path.abspath(output_file))
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=os.path.dirname(destination),
                                         prefix='.' + os.path.basename(destination) + '.',
                                         suffix='.tmp', delete=False) as dumping:
            temporary = dumping.name
            dumping.write(content)
            dumping.flush()
            os.fsync(dumping.fileno())
        if os.name != 'nt' and os.path.exists(destination):
            os.chmod(temporary, stat.S_IMODE(os.stat(destination).st_mode))
        os.replace(temporary, destination)
        temporary = None
    except (OSError, TypeError, ValueError, yaml.YAMLError) as e:
        msg = f"dumpDocDict() -> '{output_file}': {e}"
        add_error_log(msg)
        raise ValueError(msg) from e
    finally:
        if temporary is not None:
            os.unlink(temporary)


def dumpDocDict(output_file, doc, header):
    """Save one mapping with atomic replacement of one file.

    Parameters
    ----------
    output_file : str or object
        Output filename or path-like object. The parent directory must exist.
        A .json suffix selects JSON; other suffixes select YAML.
    doc : dict or list
        One mapping or a legacy list containing one mapping.
    header : str
        YAML prefix inserted verbatim; use valid YAML comment text or an empty
        string. Ignored for JSON output.

    Raises
    ------
    ValueError
        If documents cannot be normalized/serialized or the file cannot be written.

    Notes
    -----
    Returns None. YAML contains one document;
    JSON contains an object. Serialization finishes
    before replacement. Existing symlinks are followed and the resolved target is
    replaced. This is atomic for this file only, not a transaction across config files.
    """
    _dump_documents(output_file, [_single_document(_normalize_documents(doc))], header, single=True)


def dumpDocDictList(output_file, doc_list, header):
    """Save configuration mappings with atomic replacement of one file.

    Parameters
    ----------
    output_file : str or object
        Output filename or path-like object. The parent directory must exist.
        A .json suffix selects JSON; other suffixes select YAML.
    doc_list : list[dict] or dict
        Mappings to save, preserving their order.
    header : str
        YAML prefix inserted verbatim; use valid YAML comment text or an empty
        string. Ignored for JSON output.

    Raises
    ------
    ValueError
        If documents cannot be normalized/serialized or the file cannot be written.

    Notes
    -----
    Returns None. YAML uses separate documents;
    JSON contains a list. Serialization finishes
    before replacement. Existing symlinks are followed and the resolved target is
    replaced. This is atomic for this file only, not a transaction across config files.
    """
    _dump_documents(output_file, doc_list, header, single=False)
