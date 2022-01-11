The present package is designed to provide a means to calculate the nestedness of a matrix and offers a wide range of different metrics.

It is organized into two files:
- metrics.py is the main file and contains the implementation of the metrics
- deg.py is the secondary file and contains some functions necessary for the proper functioning of the main file


Some useful information about the metrics.py file:
- the basic requirement is that the input matrices are two-dimensional arrays (built with np.array)
- to ease the understanding, for each implemented metrics the article in which it was introduced is listed, which often displays a comprehensive explanation. In addition, at the end of the file some review articles, where the different metrics are explained in detail, are reported


For any doubts, questions or reporting of errors please contact the author at dario.laudati@gmail.com



# Example

import numpy as np
import metrics
import deg

M = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0]])

Nodf = metrics.NODF(M)
