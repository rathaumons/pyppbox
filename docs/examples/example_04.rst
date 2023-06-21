Example 4: Set the Main Detector/Tracker/Reider by a YAML or JSON File
======================================================================

- **Description**: Use a custom YAML or JSON file to set the main detector/tracker/reider. By doing so, you can set or adjust the parameters of each main module easily without changing your code.
- **Featuring**: 
   - :py:meth:`pyppbox.standalone.setMainDetector`
   - :py:meth:`pyppbox.standalone.setMainTracker`
   - :py:meth:`pyppbox.standalone.setMainReIDer`
   - :py:func:`pyppbox.standalone.detectPeople`
   - :py:func:`pyppbox.standalone.trackPeople`
   - :py:func:`pyppbox.standalone.reidPeople`
   - :py:func:`pyppbox.utils.visualizetools.visualizePeople`

.. literalinclude:: ../../examples/example_04_module_yaml_file.py
   :encoding: latin-1
