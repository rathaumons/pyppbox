# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os


from pyppbox.utils.logtools import ignore_this_logger, add_error_log


class ImageClass():
    """Store one training identity and its image paths.

    Attributes
    ----------
    name : str
        Identity directory name.
    image_paths : list[str]
        Supplied path list, retained by reference. Its length is the class size.
    """

    def __init__(self, name, image_paths):
        """Store a class name and its paths without opening images.

        Parameters
        ----------
        name : str
            Identity name.
        image_paths : list[str]
            Paths belonging to this identity; stored without copying.
        """
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return f"{self.name}, {len(self.image_paths)} images"
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path):
    """Read immediate identity subdirectories in sorted name order.

    Parameters
    ----------
    path : str
        Training directory; a leading user-home marker is expanded.

    Returns
    -------
    list[ImageClass]
        One entry per subdirectory. Entries inside each identity directory
        are not sorted or filtered; keep those directories image-only.

    Notes
    -----
    Reads directory listings, not image pixels. Filesystem errors propagate.
    """
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        datadir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(datadir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths_and_labels(dataset):
    """Flatten class paths and assign zero-based labels in dataset order.

    Parameters
    ----------
    dataset : list[ImageClass]
        Ordered identity classes.

    Returns
    -------
    list[str]
        Paths in class order, preserving each class's internal path order.
    list[int]
        Corresponding class indices, with one label per path.
    """
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths(datadir):
    """List paths to every directory entry without image-type filtering.

    Parameters
    ----------
    datadir : str
        Identity directory containing only training images.

    Returns
    -------
    list[str]
        Joined entry paths in filesystem listing order, or an empty list if
        the input is not a directory. Entries can include non-image files or
        subdirectories; image decoding occurs later during training.
    """
    image_paths = []
    if os.path.isdir(datadir):
        images = os.listdir(datadir)
        image_paths = [os.path.join(datadir, img) for img in images]
    return image_paths

def deepreid_extractor(model_name, model_dir, model_path, device='cuda'):
    """Construct a Torchreid feature extractor and load its model weights.

    Parameters
    ----------
    model_name : str
        Base architecture name understood by the backend.
    model_dir : str
        Directory containing base model weights.
    model_path : str
        Trained embedding-model weights.
    device : str
        Defaults to ``'cuda'``. Backend device, for example ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    object
        Initialized ``pyppbox_torchreid.utils.FeatureExtractor``.

    Raises
    ------
    ValueError
        If feature-extractor construction fails. Backend import errors propagate
        directly because the import precedes the construction error handler.
    """
    ignore_this_logger("torchreid")
    ignore_this_logger("pyppbox_torchreid")
    from pyppbox_torchreid.utils import FeatureExtractor
    extractor = []
    try:
        extractor = FeatureExtractor(
            base_model_name = model_name,
            base_model_dir = model_dir,
            model_path = model_path,
            device = device,
            verbose = False
        )
    except Exception as e:
        msg = f"deepreid_extractor() -> {e}"
        add_error_log(msg)
        raise ValueError(msg)
    return extractor
