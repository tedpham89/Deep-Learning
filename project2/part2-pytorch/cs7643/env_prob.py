""" 			  		 			     			  	   		   	  			  	
Utiliy functions.  
-----do not edit anything above this line---
"""
import time, os

from models.cnn import hello_do_you_copy as cnn_hello_do_you_copy
from models.my_model import hello_do_you_copy as mm_hello_do_you_copy
from models.resnet import hello_do_you_copy as res_hello_do_you_copy
from models.twolayer import hello_do_you_copy as tl_hello_do_you_copy
from losses.focal_loss import hello_do_you_copy as fc_hello_do_you_copy


def say_hello_do_you_copy(drive_path: str) -> None:
  print('---------- Models ------------------')
  cnn_hello_do_you_copy()
  mm_hello_do_you_copy()
  res_hello_do_you_copy()
  tl_hello_do_you_copy()


  cnn_path = os.path.join(drive_path, 'models', 'cnn.py')
  mm_path = os.path.join(drive_path, 'models', 'my_model.py')
  res_path = os.path.join(drive_path, 'models', 'resnet.py')
  tl_path = os.path.join(drive_path, 'models', 'twolayer.py')

  cnn_edit_time = time.ctime(os.path.getmtime(cnn_path))
  mm_edit_time = time.ctime(os.path.getmtime(mm_path))
  res_edit_time = time.ctime(os.path.getmtime(res_path))
  tl_edit_time = time.ctime(os.path.getmtime(tl_path))


  print('cnn.py last edited on %s' % cnn_edit_time)
  print('my_model.py last edited on %s' % mm_edit_time)
  print('resnet.py last edited on %s' % res_edit_time)
  print('twolayer.py last edited on %s' % tl_edit_time)

  print()
  print('---------- Losses ------------------')

  fc_hello_do_you_copy()

  fc_path = os.path.join(drive_path, 'losses', 'focal_loss.py')
  fc_edit_time = time.ctime(os.path.getmtime(fc_path))
  print('focal_loss.py last edited on %s' % fc_edit_time)
