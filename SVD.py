# coding:utf-8
__author__ = 'devin'

from numpy import *

"""
M = U*sigma*VT
"""
U, sigma, VT = linalg.svd([[1, 1], [7, 7]])
print U, sigma, VT