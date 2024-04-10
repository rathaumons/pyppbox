Example 8: Use real-time evaluation tools
=========================================

- **Description**: Use :class:`MyEVA` to evaluate result in real-time.
- **Featuring**: 
   - :py:class:`MyEVA` | :py:class:`pyppbox.utils.evatools.MyEVA`
   - :py:meth:`setMainDetector` | :py:meth:`pyppbox.standalone.setMainDetector`
   - :py:meth:`setMainTracker` | :py:meth:`pyppbox.standalone.setMainTracker`
   - :py:meth:`setMainReIDer` | :py:meth:`pyppbox.standalone.setMainReIDer`
   - :py:func:`detectPeople` | :py:func:`pyppbox.standalone.detectPeople`
   - :py:func:`detectPeople` | :py:func:`pyppbox.standalone.trackPeople`
   - :py:func:`reidPeople` | :py:func:`pyppbox.standalone.reidPeople`
   - :py:func:`visualizePeople` | :py:func:`pyppbox.utils.visualizetools.visualizePeople`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

.. literalinclude:: ../../examples/example_08_evatools.py
   :encoding: latin-1
