##将一个文件夹中的数据分到几个不同的子文件夹中

#!/bin/bash 

moved=0
target=yellow_double
for i in `ls yellow_double`
do
    if [ $moved -lt 2000 ]; then
	target=yellow_double
    else
	echo "moved=$moved"
	echo "break"
	break
    fi

    #echo "i=$i"
    #echo "target=$target"
    mv yellow_double/$i ./val/$target
    moved=$(($moved+1))
done

