Example 1: Use some basic standalone functions
==============================================

- **Description:** Use some standalone functions to detect people in an image, to save the detected people as an output image, to re-identify those people, and to visualize the people with the internal configurations of pyppbox.
- **Featuring:** 
   - :py:meth:`setMainDetector` | :py:meth:`pyppbox.standalone.setMainDetector`
   - :py:func:`detectPeople` | :py:func:`pyppbox.standalone.detectPeople`
   - :py:func:`setMainReIDer` | :py:func:`pyppbox.standalone.setMainReIDer`
   - :py:func:`reidPeople` | :py:func:`pyppbox.standalone.reidPeople`
   - :py:func:`visualizePeople` | :py:func:`pyppbox.utils.visualizetools.visualizePeople`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

.. literalinclude:: ../../examples/example_01_basic.py
   :encoding: latin-1

