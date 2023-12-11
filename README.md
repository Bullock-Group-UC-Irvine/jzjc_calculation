# $j_{\rm z}/j_{\rm c}$ and $j/j_{\rm c}$
Scripts for saving hdf5 files with $j_{\rm z}/j_{\rm c}$ and $j/j_{\rm c}$ for a given FIRE galaxy

## Using `jzjc.py`
`jzjc.calculation` is an executable script. Navigate to the `jzjc_calculation` folder. Then the user can execute

~~~
python jzjc.py --help
~~~

to see the details of how to use it. As an example of its most basic useage, `python jzjc.py m12i` would generate the $j_{\rm z}/j_{\rm c}$ and $j/j_{\rm c}$ for m12i in the directory specified in the users `jzjc_calculation/paths.py` file.

## The user *must* have a `paths.py` file.
This file should not be added to the git repo but should be in the `jzjc_calculation` folder. It directs the repository's scripts on the aspects of the file struture that are unique to each user. For example, to tell the repository where to save the results of `jzjc.py`, `paths.py` should contain

~~~
# paths.py

data = '/data17/grenache/your_username/jzjc_results'
~~~
