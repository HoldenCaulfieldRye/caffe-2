import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
from datetime import date
from shutil import rmtree
import copy
import json, yaml, random
import setup_data

# Warning! duplicates will be created in val and test as well, you don't want that

def main(data_dir, data_info, to_dir, target_bad_min):
  ''' This is the master function. 
  data_dir: where raw data is. data_info: where to store .txt files. '''
  All = setup_data.get_label_dict(data_dir)
  total_num_images = All.pop('total_num_images')
  Keep = setup_data.classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = setup_data.default_class(All, Keep)
  total_num_check = sum([len(Keep[key]) for key in Keep.keys()])
  if total_num_images != total_num_check:
    print "\nWARNING! started off with %i images, now have %i distinct training cases"%(total_num_images, total_num_check)
  Keep, num_output = setup_data.merge_classes(Keep)
  Keep, num_output = setup_data.check_mutual_exclusion(Keep, num_output)
  print "target bad min: %s" %(target_bad_min)
  for key in Keep.keys():
    assert len(Keep[key]) == len(set(Keep[key]))

  D = split_dict_for_cross_val(Keep)

  for key in D['train'].keys():
    assert len([el for el in D['train'][key] if el in D['val'][key]]) == 0

  # only train gets images copies
  D['train'] = rebalance_oversample(D['train'], target_bad_min)
  print 'finished rebalancing'

  # for subdict in D.keys():
  #   for key in D[subdict].keys():
  #     D[subdict][key] = copy.copy(D[subdict][key])

  # for key in D['train'].keys():
  #   assert len([el for el in D['train'][key] if el in D['val'][key]]) == 0

  # for smth in ['train','val','test']:
  #   D[smth] = setup_data.within_class_shuffle(Keep)
  # print 'finished shuffling'

  # for subdict in D.keys():
  #   for key in D[subdict].keys():
  #     D[subdict][key] = copy.copy(D[subdict][key])

  for key in D['train'].keys():
    assert len([el for el in D['train'][key] if el in D['val'][key]]) == 0

  Dump = symlink_dataset_oversample(D, data_dir, to_dir)

  assert len([el for el in Dump['train'] if el in Dump['val']]) == 0

  if data_info is not None:
    dump_to_files_oversample(Keep, Dump, data_info)
  return num_output, Dump


def split_dict_for_cross_val(Keep):
  D = {'train':{}, 'val':{}, 'test':{}}
  part = [0, 0.8, 0.87, 1] # partition into train val test
  for i,smth in enumerate(['train','val','test']):
    for key in Keep.keys():
      #assert len(Keep[key]) == len(set(Keep[key]))
      print 'Keep[%s] has no duplicates'%(key)
      l = len(Keep[key])
      print 'D[%s][%s] gets elements %i to %i'%(smth,key,int(part[i]*l),int(part[i+1]*l))
      print 'Keep[%s] starts with'%(key), Keep[key][:3]
      D[smth][key] = copy.copy(Keep[key][int(part[i]*l):int(part[i+1]*l)])
      if smth != 'train':
        a = [el for el in D[smth][key] if el in D['train'][key]]
        print "%i els in D['val'][%s] are also in D['train'][%s]"%(len(a),key,key)
        #assert len(a) == 0
      # random.shuffle(D[smth][key])
      print ''
  for smth in ['train','val','test']:
    for key in D[smth].keys():
      #assert len(D[smth][key]) == len(set(D[smth][key]))
      print 'D[%s][%s] has no duplicates'%(smth,key)
      # print 'D[%s][%s] has %i elements'%(smth,key,len(D[smth][key]))
    print ''
  for key in D['val'].keys():
    a = [el for el in D['val'][key] if el in D['train'][key]]
    print "%i els in D['val'][%s] are also in D['train'][%s]"%(len(a),key,key)
    #assert len(a) == 0
  return D


def rebalance_oversample(Keep, target_bad_min):
  '''if target_bad_min not given, prompts user for one; 
  and implements it. Note that with >2 classes, this can be 
  implemented either by downsizing all non-minority classes by the
  same factor in order to maintain their relative proportions, or 
  by downsizing as few majority classes as possible until
  target_bad_min achieved. We can assume that we care mostly about 
  having as few small classes as possible, so the latter is 
  implemented.'''
  if target_bad_min == 'N': return Keep
  else: target_bad_min = float(target_bad_min)
  # minc is class with minimum number of training cases
  ascending_classes = sorted([(key,len(Keep[key]))
                              for key in Keep.keys()],
                             key=lambda x:x[1])
  maxc, len_maxc = ascending_classes[-1][0], ascending_classes[-1][1]
  minc, len_minc = ascending_classes[0][0], ascending_classes[0][1]
  total_num_images = sum([len(Keep[key]) for key in Keep.keys()])
  # print ascending_classes
  # print "\ntotal num images: %i"%(total_num_images)
  minc_proportion = float(len_minc)/total_num_images
  maxc_proportion = float(len_maxc)/total_num_images
  if target_bad_min is None:
    target_bad_min = raw_input("\nmax class currently takes up %.2f, what's your target? [num/N] "%(maxc_proportion))
  if target_bad_min is not 'N':
    target_bad_min = float(target_bad_min)
    print 'minc_proportion: %.2f, target_bad_min: %.2f'%(minc_proportion, target_bad_min)
    if maxc_proportion > target_bad_min:
      copy_size = ( target_bad_min*(total_num_images) - len_minc ) / (1 - target_bad_min)
      copy_size = int(copy_size)
      random.shuffle(Keep[minc])
      print '%s has %i images so %i copies will be made'%(minc, len_maxc,copy_size)
      min_imgs_copy = copy.copy(Keep[minc])
      print 'min_imgs_copy has %i images'%(len(min_imgs_copy))
      # number_of_copies = 
      # number_of_copies = int(number_of_copies)
      for i in range((copy_size/len_minc)):
        Keep[minc] += min_imgs_copy
      Keep[minc] += min_imgs_copy[:(copy_size % len_minc)]
      random.shuffle(Keep[minc])
      total_num_images = len(Keep[minc]) + len(Keep[maxc])
      print 'minc now has %i images, compared to %i for maxc'%(len(Keep[minc]), len(Keep[maxc]))
      assert len(Keep[maxc])/float(total_num_images) == target_bad_min
  return Keep


def symlink_dataset_oversample(D, from_dir, to_dir):
  Dump = {}
  if os.path.isdir(to_dir): rmtree(to_dir)
  os.mkdir(to_dir)
  for dname in ['train','val','test']:
    Dump[dname] = []
    for [num,key] in enumerate(D[dname].keys()):
      Dump[dname] += [[f,num] for f in D[dname][key]]
    random.shuffle(Dump[dname])
    data_dst_dir = ojoin(to_dir,dname)
    os.mkdir(data_dst_dir)
    for i in xrange(len(Dump[dname])):
      if os.path.islink(ojoin(data_dst_dir,Dump[dname][i][0])): 
        if dname == 'train':
          old = Dump[dname][i][0]
          while os.path.islink(ojoin(data_dst_dir,Dump[dname][i][0])):
            Dump[dname][i][0]=Dump[dname][i][0].split('.')[0]+'_.jpg'
          os.symlink(ojoin(from_dir,old),
                     ojoin(data_dst_dir,Dump[dname][i][0]))
        else: continue
      else: os.symlink(ojoin(from_dir,Dump[dname][i][0]),
                       ojoin(data_dst_dir,Dump[dname][i][0]))
  return Dump


def dump_to_files_oversample(Keep, Dump, data_info):
  if os.path.exists(data_info): rmtree(data_info)
  os.mkdir(data_info)
  assert len([el for el in Dump['train'] if el in Dump['val']]) == 0
  for key in Dump.keys():
    dfile = open(ojoin(data_info,key+'.txt'),'w')
    dfile.writelines(["%s %i\n"%(f,num) for (f,num) in Dump[key]])
    dfile.close()
    
  # write to read file how to interpret values as classes      
  read_file = open(ojoin(data_info,'read.txt'), 'w')    
  read_file.writelines(["%i %s\n" % (num,label) for (num, label)
                         in enumerate(Keep.keys())])
  read_file.close()



if __name__ == '__main__':
  import sys

  print "Warning! duplicates will be created in val and test as well, you don't want that"
  
  target_bad_min, data_dir, data_info = None, None, None
  for arg in sys.argv:
    if "bad-min=" in arg:
      target_bad_min = arg.split('=')[-1]
      print "target bad min: %s" %(target_bad_min)
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "to-dir=" in arg:
      to_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])

  # data_dir = /data/ad6813/pipe-data/Bluebox/raw_data/dump
  # data_info = /data/ad6813/caffe/data_info
  if data_dir is None:
    print "\nERROR: data_dir not given"
    exit
      
  # careful, unlike setup.py Dump is a dict
  num_output, Dump = main(data_dir, data_info, to_dir, target_bad_min)
  print "\nIt's going to say 'An exception has occured etc'"
  print "but don't worry, that's num_output info for the training shell script to use\n"
  sys.exit(num_output)
