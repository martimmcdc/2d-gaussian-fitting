#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from visual_methods import *
from functions import *
from automation import *

table = input("Name of table file: ")
files = ["all"]
path = input("Name of data folder: ")
save_path = input("Name of destination folder: ")
wavelength = input("Wavelength to use from table (in case multiple are stated): ")
show_int = int(input("Show results plots? 1-yes, 0-no"))
fit_rad = float(input("Fitting radius (by default 4): "))

var_pos = 0.001
if show_int==1:
    show = True
elif show_int==0:
    show = False

new_files = automation(table=table,files=files,path=path,save_path=save_path,
                       wavelength=wavelength,show=show,var_pos=var_pos,fitting_radius=fit_rad)
    
print("New file names: \n\n",new_files)