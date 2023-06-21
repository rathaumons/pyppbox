Example 1: Use Some Basic Standalone Functions
==============================================

- **Description:** Use some standalone functions to detect people in an image, to save the detected people as an output image, to re-identify those people, and to visualize the people with the internal configurations of pyppbox.
- **Featuring:** 
   - :py:meth:`pyppbox.standalone.setConfigDir`
   - :py:func:`pyppbox.standalone.detectPeople`
   - :py:func:`pyppbox.standalone.reidPeople`
   - :py:func:`pyppbox.utils.visualizetools.visualizePeople`

.. literalinclude:: ../../examples/example_01_basic.py
   :encoding: latin-1
