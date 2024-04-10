Example 5: Set the main modules by a raw/ready dictionary
=========================================================

- **Description**: Use a raw/ready dictionary to set the main detector/tracker/reider. By doing so, you can set or adjust the parameters of each main module directly in codes.
- **Featuring**: 
   - :py:meth:`setMainDetector` | :py:meth:`pyppbox.standalone.setMainDetector`
   - :py:meth:`setMainTracker` | :py:meth:`pyppbox.standalone.setMainTracker`
   - :py:meth:`setMainReIDer` | :py:meth:`pyppbox.standalone.setMainReIDer`
   - :py:func:`detectPeople` | :py:func:`pyppbox.standalone.detectPeople`
   - :py:func:`trackPeople` | :py:func:`pyppbox.standalone.trackPeople`
   - :py:func:`reidPeople` | :py:func:`pyppbox.standalone.reidPeople`
   - :py:func:`visualizePeople` | :py:func:`pyppbox.utils.visualizetools.visualizePeople`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

.. literalinclude:: ../../examples/example_05_module_json_str_dict.py
   :encoding: latin-1
