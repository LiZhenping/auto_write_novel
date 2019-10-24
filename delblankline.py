# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:57:34 2019

@author: lizhenping
"""

def delblankline(infile, outfile):
 infopen = open(infile, 'r')
 outfopen = open(outfile, 'w')
 
 lines = infopen.readlines()

 for line in lines:

  if line.split():
   line = ''.join(line.split())
   
   outfopen.writelines(line)
  else:
   outfopen.writelines("\n")
 
 infopen.close()
 outfopen.close()
 
delblankline("four.txt", "four1.txt")
