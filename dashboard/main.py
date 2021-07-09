""" Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve sliders.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/sliders

in your browser.

"""

# from controller import Controller
import controller
from bokeh.io import curdoc

controller.Controller(curdoc)
