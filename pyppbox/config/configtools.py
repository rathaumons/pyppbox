# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import json
import os
import stat
import tempfile
from pathlib import Path
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


def _serialize_documents(output_file, documents, header, single):
    documents = _normalize_documents(documents)
    if os.path.splitext(os.fspath(output_file))[1].lower() == '.json':
        value = _single_document(documents) if single else documents
        return json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    content = yaml.safe_dump_all(documents, sort_keys=False, allow_unicode=True)
    if header:
        content = header + ('' if header.endswith('\n') else '\n') + content
    return content


def _dump_documents(output_file, documents, header, single):
    temporary = None
    try:
        content = _serialize_documents(output_file, documents, header, single)
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


def _sync_bytes(path, content):
    with open(path, 'wb') as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _clean_save(directory):
    # Only files created in our transaction directory; never recurse into a tree.
    # Once cleanup starts, originals have been restored or the new group committed.
    # Remove the journal first so interrupted cleanup cannot request another rollback.
    (directory / 'manifest.json').unlink(missing_ok=True)
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()


def _restore_save(directory, entries):
    errors = []
    for index, entry in enumerate(entries):
        destination = entry['destination']
        try:
            if entry['existed']:
                temporary = directory / f'{index}.restore'
                _sync_bytes(temporary, (directory / f'{index}.backup').read_bytes())
                if os.name != 'nt':
                    os.chmod(temporary, entry['mode'])
                os.replace(temporary, destination)
            elif os.path.exists(destination):
                os.unlink(destination)
        except OSError as error:
            errors.append(f'{destination}: {error}')
    if errors:
        raise ValueError(f'Config rollback incomplete; retain recovery files in {directory}: ' + '; '.join(errors))
    _clean_save(directory)


def _dump_config_files(cfg_dir, files):
    """Stage a group, retaining original bytes for rollback or explicit recovery."""
    root = Path(cfg_dir).resolve()
    _check_pending_save(root)
    directory = None
    entries = []
    ready = False
    committed = False
    try:
        # Finish all serialization before creating the transaction or changing configs.
        rendered = [(name, _serialize_documents(name, docs, header, single).encode('utf-8'))
                    for name, docs, header, single in files]
        directory = Path(tempfile.mkdtemp(prefix='.pyppbox-save-', dir=root))
        for index, (name, content) in enumerate(rendered):
            destination = os.path.realpath(root / name)
            if destination in [entry['destination'] for entry in entries]:
                raise ValueError('Config files must resolve to distinct destinations.')
            existed = os.path.exists(destination)
            mode = stat.S_IMODE(os.stat(destination).st_mode) if existed else 0o600
            if existed:
                _sync_bytes(directory / f'{index}.backup', Path(destination).read_bytes())
            staged = directory / f'{index}.staged'
            _sync_bytes(staged, content)
            if os.name != 'nt':
                os.chmod(staged, mode)
            entries.append(dict(name=name, destination=destination, existed=existed, mode=mode))
        _sync_bytes(directory / 'manifest.json', json.dumps(entries).encode('utf-8'))
        ready = True
        for index, entry in enumerate(entries):
            os.replace(directory / f'{index}.staged', entry['destination'])
        _sync_bytes(directory / 'committed', b'complete\n')
        committed = True
        _clean_save(directory)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        if directory is not None and directory.exists() and not committed:
            if ready:
                _restore_save(directory, entries)
            else:
                _clean_save(directory)
        raise ValueError(f'Cannot save configuration group: {error}') from error


def _check_pending_save(cfg_dir):
    pending = next(Path(cfg_dir).glob('.pyppbox-save-*'), None)
    if pending is not None:
        raise ValueError(f'A config save needs recovery or is still running: {pending}')


def recoverConfigSave(cfg_dir):
    """Recover an interrupted ``MyConfigurator.dumpConfigs()`` save.

    Parameters
    ----------
    cfg_dir : str or object
        Config directory containing the retained ``.pyppbox-save-*`` folder.
        Call only when no process is saving or using this directory. Pending saves
        restore original bytes; completed saves retain the new files.

    Raises
    ------
    ValueError
        Recovery metadata is invalid, a destination changed, or restoration failed.
        Backups are retained on failure. A group save is recoverable, not atomic to
        concurrent readers; individual file replacements remain atomic. Power-loss
        durability depends on the filesystem. Returns None.
    """
    root = Path(cfg_dir).resolve()
    for directory in root.glob('.pyppbox-save-*'):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f'Invalid config recovery directory: {directory}')
        try:
            manifest = directory / 'manifest.json'
            if not manifest.exists() or (directory / 'committed').exists():
                _clean_save(directory)
                continue
            entries = json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(entries, list):
                raise ValueError('Invalid config recovery manifest.')
            for entry in entries:
                if entry['name'] not in ('main.yaml', 'detectors.yaml', 'trackers.yaml', 'reiders.yaml'):
                    raise ValueError('Invalid config recovery filename.')
                if os.path.realpath(root / entry['name']) != entry['destination']:
                    raise ValueError('Config recovery destination changed.')
            _restore_save(directory, entries)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Cannot recover configuration save in {directory}: {error}') from error
