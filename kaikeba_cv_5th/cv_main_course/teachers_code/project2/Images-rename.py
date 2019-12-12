import os

class ImageRename():
    def __init__(self,r,s):
        self.path = r+ '/' + s
    def rename(self):
        filelist = os.listdir(self.path)
        # os.listdir(path) 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        # path -- 需要列出的目录路径
        total_num = len(filelist)

        n = 0
        for item in filelist:
            if item.endswith('.jpg'):
                oldname = os.path.join(os.path.abspath(self.path), item)
                # os.path.abspath(path) 返回绝对路径
                #os.path.join()函数：连接两个或更多的路径名组件
                #如果各组件名首字母不包含’ / ’，则函数会自动加上
                # 如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
                # 如果最后一个组件为空，则生成的路径以一个’ / ’分隔符结尾
                newname = os.path.join(os.path.abspath(self.path), s+ format(str(n), '0>3s') + '.jpg')
                os.rename(oldname, newname)
                n += 1
        print ('total %d to rename & converted %d jpgs' % (total_num, n))

PHASE=['train','val']
SPECIES = ['rabbits', 'rats', 'chickens']

for p in PHASE:
    for s in SPECIES:
        newname = ImageRename(p,s)
        newname.rename()