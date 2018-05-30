# cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
# root_dir=$cur_dir/../..

# cd $root_dir
export PYTHONPATH=$PYTHONPATH:/media/e/FrameWork/caffe-ssd/python

redo=1
data_root_dir="./ssd-face"
dataset_name="realface"
mapfile="./labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train
do
  python /media/e/FrameWork/caffe-ssd/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile\
   --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height\
   --check-label $extra_cmd $data_root_dir ./$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db\
  ./ssd-face/examples/$dataset_name
done
  #  examples/$dataset_name
