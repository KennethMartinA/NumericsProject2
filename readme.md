# Introduction
This is a finiteElement solver for a triangular mesh. It is primarily meant to run an advection diffusion solver over a triangular mesh domain that is present in data/, but it has the flexibility to work with other domains and equations, though that latter will require getting your hands dirty with the code.

# Setup

You can first try activating the venv, by running venv/scripts/activate.bat

If the venv environment does not work, or this is not an option, you will need to install the local package finite-element.

Run

> python pip install -m finite-element

Next, you will need to run finiteElement/SpatialInterpolator.py to create model.pkl (which unfortunately is too big to push to github). This will store a local file that includes parameters for the spatiotemporal interpolator the FEM model uses.

>python finiteElement/SpatialInterpolator.py [-h]

You can run the file directly to create model.pkl.

Finally, you can run finiteElement/main.py.

> python finiteElement/main.py.

Though it should be workable with the default, if you need help on the args available for main.py, you can run

> python finiteElement/main.py -h

You can also run main.bat in the top-level folder, which also contains a representative sample of args,
and is how the main data used for the report was obtained.
# Structure

The directory finiteElement contains the main code necessary for everything.
In particular
    * utils.py: contains the source function and the spatial psi filtering function
    * constant.py: contains important locations of constants
    * Element.py: contains the main classes builduing up the two types of elements, Element (for steady state diffusion, but not implemented to run from a script) and AdvecDiffElement
    * Space.py: Contains the main space classes, in particular sparseAdvecDiffSpace, which contains the methods for assembling the elements and timestepping the advection diffusion 
    * SpatialInterpolator.py: 

# Wind Interpolation
The wind data was obtained from the Midas OPEN: UK Mean Wind Dataset[^1], with many thanks to Nyall Oswald who was able to figure out how to download, process and clean the data. The Spatial Interpolator uses a Matern kernel for the interpolation, but other options might work effectively too. Unfortunately, it is not very customizable and requires tinkering inside the class itself.


[^1]: Met Office (2024): MIDAS Open: UK mean wind data, v202407. NERC EDS Centre for Environmental Data Analysis, 06 August 2024. doi:10.5285/91cb9985a6c2453d99084bde4ff5f314. https://dx.doi.org/10.5285/91cb9985a6c2453d99084bde4ff5f314