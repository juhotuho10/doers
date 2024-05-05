# doers
[![Codecov](https://codecov.io/github/juhotuho10/doers/coverage.svg?branch=master)](https://app.codecov.io/gh/juhotuho10/doers)
[![Dependency status](https://deps.rs/repo/github/juhotuho10/doers/status.svg)](https://deps.rs/repo/github/juhotuho10/doers)

A Rust port of [pyDOE2](https://github.com/clicumu/pyDOE2)

Rust crate for generating Design Of Experiments


TODO
-------
- Finish implementing all core functionality from pyDOE
- Maybe add more newer design patters and / or other functional to help with generating experiments 


Credits
-------

original Scilab code was originally made by:    
- Copyright (C) 2012 - Maria Christopoulou
- Copyright (C) 2012 - 2013 - Michael Baudin
- Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
- Copyright (C) 2009 - Yann Collette
- Copyright (C) 2009 - CEA - Jean-Marc Martinez

- Website: [scidoe](https://atoms.scilab.org/toolboxes/scidoe/0.4.1)

converted to Python and worked on by:
- Copyright (C) 2014 - Abraham D. Lee
- git repo: [pyDOE](https://github.com/tisimst/pyDOE)

- Copyright (C) 2018 - Rickard Sjögren and Daniel Svensson
- git repo: [pyDOE2](https://github.com/clicumu/pyDOE2)

Converted to Rust and worked on by:
- Copyright (C) 2024 - Juho Naatula
- Cargo.io page: doers [to be added]

References
----------

- [Factorial designs](http://en.wikipedia.org/wiki/Factorial_experiment)
- [Plackett-Burman designs](http://en.wikipedia.org/wiki/Plackett-Burman_design)
- [Box-Behnken designs](http://en.wikipedia.org/wiki/Box-Behnken_design)
- [Central composite designs](http://en.wikipedia.org/wiki/Central_composite_design)
- [Latin-Hypercube designs](http://en.wikipedia.org/wiki/Latin_hypercube_sampling)
- Surowiec, Izabella, Ludvig Vikström, Gustaf Hector, Erik Johansson,
Conny Vikström, and Johan Trygg. “Generalized Subset Designs in Analytical
Chemistry.” Analytical Chemistry 89, no. 12 (June 20, 2017): 6491–97.
https://doi.org/10.1021/acs.analchem.7b00506.
