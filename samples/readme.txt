首先将windwos中fonts目录下的simhei.ttf拷贝到/home/hadoop/.pyenv/versions/2.7.10/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf(文件路径参考1.c，根据实际情况修改)目录中，
然后删除~/.cache/matplotlib的缓冲目录
第三在代码中动态设置参数：
[python] view plain copy

    #coding:utf-8
    import matplotlib
    matplotlib.use('qt4agg')
    #指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family']='sans-serif'
    #解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.plot([-1,2,-5,3])
    plt.title(u'中文',fontproperties=myfont)
    plt.show()

以下方法是在当前的分支下新建一个分支，然后把之前分支删除，接着把新建的分支重命名为原分支名称，最后把分支强制推送到远程

1.Checkout

   git checkout --orphan latest_branch

2. Add all the files

   git add -A

3. Commit the changes

   git commit -am "commit message"


4. Delete the branch

   git branch -D master

5.Rename the current branch to master

   git branch -m master

6.Finally, force update your repository

   git push -f origin master