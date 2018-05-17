#画图工具
processon
#文档撰写
office online

tail -f filename #实时显示文件最新内容
:%s/^/... #文件头添加
: s/vivian/sky/ #替换当前行第一个
: s/vivian/sky/g #替换当前行所有
: n,$s/vivian/sky/ #替换第n行到最后一行第一个
: n,$s/vivian/sky/g #替换第n行到最后一行所有

script filename #终端信息保存到文件中
env | grep -i proxy #查看代理
%!python -m json.tool #终端格式化json文件
paste -d ' ' 1.txt 2.txt >>merge.txt #以空格按顺序合并两个文件

sed -i "15i contents" Lab.txt #选中第几行插入内容
sed -n "1,21p" Lab.txt >>test.txt #从文件中选中的行写入到另一个文件中
cat train.txt | sort -k2 -n #将文件中所有项按第二个字段排序
cat train.txt | awk '{print $2}' |sort|uniq -c | sort -k2 -n #将文件第二个字段的个数进行统计，并按第二个字段排序(统计每个类别的个数)
