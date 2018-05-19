#coding=utf-8
import os,re,glob,sys,json
import numpy as np
import binascii 
import argparse
import time
# dist = np.linalg.norm(vec1 - vec2) 
import struct
import multiprocessing as mp

_BASE_DIR_ = os.path.dirname(__file__)
_INSIGHTFACE_DIR_ = os.path.join(_BASE_DIR_,'../insightFace')
_MEGAFACE_DIR_ = os.path.join(_BASE_DIR_,'../devkit')

sys.path.append(_INSIGHTFACE_DIR_)
sys.path.append(_MEGAFACE_DIR_)

sys.path.append(os.path.join(_INSIGHTFACE_DIR_, 'src','megaface'))
import gen_megaface,remove_noises

sys.path.append(os.path.join(_MEGAFACE_DIR_, 'experiments'))
import run_experiment

_MODEL_BASE_DIR_ = "/home/guoxufeng/work/models/"
_DATA_BASE_DIR_ = "/home/guoxufeng/work/data/"

sys.path.append(os.path.join(_BASE_DIR_, '..', 'insightFace', 'src' ,'common'))
import face_preprocess

_DEFAULT_CODE_ = "rxx_default_code"

# import face_image
# sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))

# import face_preprocess
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# _RESULT_DIR_ = '/home/guoxufeng/work/data/MegaFace/Arc_cm_results_felix/otherFiles/'
# _RESULT_DIR_ = os.path.join(_BASE_DIR_,'tmp_Arc_cm_results_felix_new')
# _RESULT_DIR_ = os.path.join(_BASE_DIR_,'tmp_m1_arcface_431_felix_cm')
# import random

import tempfile
tempdir = tempfile.mkdtemp(dir="/tmp",prefix="felix_")

def gen_mini_list(distractor_list_path_test,megaface_lst,probe_list_test,facescrub_lst):
  # from here, generate the corresponding mini mega face lst file
  if not os.path.exists(distractor_list_path_test):
    os.makedirs(distractor_feature_path)

  facescrub_dict = {"path":[],"id":[]}
  with open(facescrub_lst,'r') as f:
    for iLine in f.readlines():
      fullPath = iLine.split("\t")[1]
      idPath = "/".join(fullPath.split('/')[-2:])
      personid = fullPath.split('/')[-2]
      facescrub_dict['path'].append(idPath)
      facescrub_dict['id'].append(personid)
  with open(probe_list_test,'w') as f:
    f.write(json.dumps(facescrub_dict))

  mega_dict = {"path":[]}
  with open(megaface_lst,'r') as f:
    for iLine in f.readlines():
      fullPath = iLine.split("\t")[1]
      idPath = "/".join(fullPath.split('/')[-3:])
      # personid = fullPath.split('/')[-3]
      # print "/".join(idPath)
      mega_dict['path'].append(idPath)
  with open(os.path.join(distractor_list_path_test,'megaface_features_list.json_10_1'),'w') as f:
    f.write(json.dumps(mega_dict))


def gen_not_done_list(args,input_lst, lst_type):
  print input_lst
  input_lst_basename = os.path.basename(input_lst)
  input_lst_remains = os.path.join(tempdir,input_lst_basename+'.0')
  for i in range(1,10):
    if os.path.exists(input_lst_remains):
      input_lst_remains = os.path.join(tempdir,input_lst_basename+'.%d'%(i))
    else:
      break

  with open(input_lst_remains, "w") as f2:
      

    for line in open(input_lst, 'r'):
      # if i%10000==0:
      #   print("writing mf",i, succ)
      # i+=1
      # if i<=args.skip:
      #   continue
      image_path, label, bbox, landmark, aligned = face_preprocess.parse_lst_line(line)
      # assert aligned==True

      if lst_type == "mega":
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(args.fea_mega_out, a1, a2)
        out_path = os.path.join(out_dir, b+"_arcface_112x112.bin")
      elif lst_type == "scrub":
        _path = image_path.split('/')
        a,b = _path[-2], _path[-1]
        #a = a.replace(' ', '_')
        #b = b.replace(' ', '_')
        out_dir = os.path.join(args.fea_scrub_out, a)
        out_path = os.path.join(out_dir, b+"_arcface_112x112.bin")
      if not os.path.exists(out_path):
        f2.write(line)
  return input_lst_remains
def prepare_para_lst_list(megaface_lst, facescrub_lst, splits):
  

  print "tempdir: ",tempdir

  # print facescrub_lst
  # print megaface_lst
  num_lines_megaface_lst = sum(1 for line in open(megaface_lst))
  print "num_lines_megaface_lst: ",num_lines_megaface_lst


  num_lines_facescrub_lst = sum(1 for line in open(facescrub_lst))
  print "num_lines_facescrub_lst: ",num_lines_facescrub_lst


  emptyLstPath = os.path.join(tempdir,'empty.lst')
  with open(emptyLstPath, 'w') as f:
    f.write("") 

  lstFileDictList = []
  if num_lines_facescrub_lst == 0:
    mega_splits = splits
  else:
    lstFileDictList.append({
      "mega_lst":emptyLstPath,
      "scrub_lst":facescrub_lst
      })
    mega_splits = splits -1 

  cutMega = num_lines_megaface_lst / mega_splits
  print cutMega

  
  for indexProcess in range(mega_splits):
    iMegaSubLstPath = os.path.join(tempdir,'mega_sub_%02d.lst'%(indexProcess))

    print iMegaSubLstPath
    startIndex = indexProcess*cutMega
    
    if indexProcess == mega_splits -1:
      endIndex = -1
      with open(megaface_lst) as myfile:
        nLines=myfile.readlines()[startIndex:] 
    else:
      endIndex = (1+indexProcess)*cutMega
      with open(megaface_lst) as myfile:
        nLines=myfile.readlines()[startIndex:endIndex] 

    
    with open(iMegaSubLstPath, "w") as f2:
      for iLine in nLines:
          f2.write(iLine)
    lstFileDictList.append({
      "mega_lst":iMegaSubLstPath,
      "scrub_lst":emptyLstPath
      })
    print startIndex,endIndex,num_lines_megaface_lst
  return lstFileDictList




def main(args):
  print "============================"
  for k,v in  vars(args).iteritems():
    print k,": ",v
  print "============================"
  lstfile = "lst"
  testListDir = "templatelists"
  if args.mini:
    lstfile = "mini_lst"
    testListDir = 'templatelists_mini'
    print "using mini lst file"

  facescrub_lst = _DATA_BASE_DIR_ + '/MegaFace/112x112/'+lstfile
  megaface_lst=_DATA_BASE_DIR_ + '/MegaFace/megaface_aligned/'+lstfile

  distractor_list_path_test = os.path.join(_MEGAFACE_DIR_, testListDir)
  probe_list_test = os.path.join(_MEGAFACE_DIR_, testListDir, 'facescrub_features_list.json')   

  if args.mini:
    gen_mini_list(distractor_list_path_test,megaface_lst,probe_list_test,facescrub_lst)

  gpuList = args.gpus.split(',')
  print "gpuList to calculate: ",gpuList



  megaface_lst_remains = gen_not_done_list(args,megaface_lst,'mega')
  facescrub_lst_remains = gen_not_done_list(args,facescrub_lst,'scrub')
  num_facescrub_lst_remains = sum(1 for line in open(facescrub_lst_remains))
  print facescrub_lst_remains,num_facescrub_lst_remains

  isFeatureChanged = False
  if len(gpuList) > 1:
    lstFileDictList = prepare_para_lst_list(megaface_lst_remains, facescrub_lst_remains, len(gpuList))
    processes = []
    for indexProcess,iGPU in enumerate(gpuList):
      iLstFileDict = lstFileDictList[indexProcess]
      args_gen_megaface = argparse.Namespace(
        algo='arcface', 
        batch_size=100, 
        concat=0, 
        facescrub_lst=iLstFileDict['scrub_lst'],  #sub list
        facescrub_out=args.fea_scrub_out, 
        fsall=0, 
        gpu=int(iGPU), 
        image_size='3,112,112', 
        mean=0, 
        megaface_lst=iLstFileDict['mega_lst'], # sub list
        megaface_out=args.fea_mega_out, 
        mf=1, 
        model=args.test_model, 
        seed=727, 
        skip=0)

      # return 0
      num_lines_megaface_lst = sum(1 for line in open(iLstFileDict['mega_lst']))

      num_lines_facescrub_lst = sum(1 for line in open(iLstFileDict['scrub_lst']))
      if num_lines_megaface_lst + num_lines_facescrub_lst >0 :
        processes.append(mp.Process(target=gen_megaface.main, args=(args_gen_megaface,)))
      # gen_megaface.main(args_gen_megaface)


    # Run processes
    for p in processes:
      isFeatureChanged = True
      p.start()
      pass

    # # Exit the completed processes
    for p in processes:
      p.join()

  megaface_lst_remains = gen_not_done_list(args,megaface_lst_remains,'mega')

  num_lines_megaface_lst = sum(1 for line in open(megaface_lst_remains))

  facescrub_lst_remains = gen_not_done_list(args,facescrub_lst_remains,'scrub')
  num_lines_facescrub_lst = sum(1 for line in open(facescrub_lst_remains))
  print "faces to process still remains: ",num_lines_megaface_lst + num_lines_facescrub_lst
  # print "".join([line for line in open(megaface_lst_remains)][0:10])

  if num_lines_megaface_lst + num_lines_facescrub_lst >0 :
    isFeatureChanged = True
    args_gen_megaface_full = argparse.Namespace(
        algo='arcface', 
        batch_size=100, 
        concat=0, 
        facescrub_lst=facescrub_lst,  #full list
        facescrub_out=args.fea_scrub_out, 
        fsall=0, 
        gpu=int(gpuList[0]), 
        image_size='3,112,112', 
        mean=0, 
        megaface_lst=megaface_lst_remains, # full list
        megaface_out=args.fea_mega_out, 
        mf=1, 
        model=args.test_model, 
        seed=727, 
        skip=0)

    gen_megaface.main(args_gen_megaface_full)

    megaface_lst_remains = gen_not_done_list(args,megaface_lst_remains,'mega')
    num_lines_megaface_lst = sum(1 for line in open(megaface_lst_remains))
    print "num_lines_megaface_lst still remains2: ",num_lines_megaface_lst
    print "".join([line for line in open(megaface_lst_remains)][0:10])

  # return 0


  args_remove_noises = argparse.Namespace(
    algo='', 
    facescrub_feature_dir=args.fea_scrub_out, 
    facescrub_feature_dir_out=args.cm_scrub_out, 
    facescrub_lst=facescrub_lst, 
    facescrub_noises=os.path.join(_INSIGHTFACE_DIR_, 'src','megaface','facescrub_noises.txt'), 
    megaface_feature_dir=args.fea_mega_out, 
    megaface_feature_dir_out=args.cm_mega_out, 
    megaface_lst=megaface_lst, 
    megaface_noises=os.path.join(_INSIGHTFACE_DIR_, 'src','megaface','megaface_noises.txt'), 
    suffix='arcface_112x112')
  
  if isFeatureChanged > 0:
    remove_noises.main(args_remove_noises)

  
  args_run_exp = argparse.Namespace(
    delete_matrices=False, 
    distractor_feature_path=args.cm_mega_out, 
    distractor_list_path=  distractor_list_path_test,
    file_ending='_arcface_112x112.bin', 
    model=_MEGAFACE_DIR_ + '/models/jb_identity.bin', 
    num_sets=1, 
    out_root=args.result_out, 
    probe_feature_path=args.cm_scrub_out,
    probe_list=probe_list_test, 
    sizes=[100000,1000000]
    )
  run_experiment.main(args_run_exp)
  # python run_experiment.py /home/guoxufeng/work/data/MegaFace/distractor_special_feature/ /home/guoxufeng/work/data/MegaFace/probe_special_feature/ _arcface_112x112.bin /home/guoxufeng/work/data/MegaFace/output_special_feature/ --probe_list /home/guoxufeng/work/faceRec/devkit/felix/special_feature_list/probe_list.json  --distractor_list_path /home/guoxufeng/work/faceRec/devkit/felix/special_feature_list/ --sizes  100

  pass
def parse_arguments(argv):

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # subparsers = parser.add_subparsers(help='commands')

  # base_parser = subparsers.add_parser('base', help='Basic contents')
  parser.add_argument('--code', type=str, help='suggest: <model>_<checkPoint>', default=_DEFAULT_CODE_)
  parser.add_argument('--mini', action="store_true", help="~",default=False)

  parser.add_argument('--gpus', type=str, help="~",default="4")

  # detail_parser = subparsers.add_parser('detail', help='Detail contents')
  parser.add_argument('--test-model', type=str, help='~', default=os.path.join(_MODEL_BASE_DIR_,'yzf_models','model-r100-softmax1e3/model-r100,56'))
  parser.add_argument('--fea-mega-out', type=str, help='see gen_megaface help', default=os.path.join(_DATA_BASE_DIR_,'bulkMega',"mega_fea_"+_DEFAULT_CODE_))
  parser.add_argument('--fea-scrub-out', type=str, help='see gen_megaface help', default=os.path.join(_DATA_BASE_DIR_,'bulkMega',"scrub_fea_"+_DEFAULT_CODE_))
  parser.add_argument('--cm-mega-out', type=str, help='see remove_noises help', default=os.path.join(_DATA_BASE_DIR_,'bulkMega',"mega_fea_"+_DEFAULT_CODE_+"_cm"))
  parser.add_argument('--cm-scrub-out', type=str, help='see remove_noises help', default=os.path.join(_DATA_BASE_DIR_,'bulkMega',"scrub_fea_"+_DEFAULT_CODE_+"_cm"))
  parser.add_argument('--result-out', type=str, help='see run_experiment help', default=os.path.join(_DATA_BASE_DIR_,'bulkMega',"result_"+_DEFAULT_CODE_))

  
  return parser.parse_args(argv)

if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  # print args
  _DEFAULT_CODE_ = args.code
  args = parse_arguments(sys.argv[1:])
  # print "  ==  "
  # print args
  # megaface_out = MEGA_OUT_BASE + args.output_postfix
  # facescrub_out = FACESCRUB_OUT_BASE  + args.output_postfix

  # print ("megaface_out: ",megaface_out)
  # print ("facescrub_out: ",facescrub_out)
  # print (vars(args).keys())
  main(args)


