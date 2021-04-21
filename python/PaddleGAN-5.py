#!/usr/bin/env python
# coding: utf-8

# # 全网都在求的「蚂蚁呀嘿」教程--基于PaddleGAN的First order motion model实现
# 
# 什么？你还不知道「蚂蚁呀嘿」？这位兄台，那你可能out得相当严重！
# 
# 这是引起男女老少的争相关注、火爆🔥抖音、B站、全网都在求教程的「蚂蚁呀嘿」魔法！
# 而这秘密就藏在[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)飞桨生成对抗网络套件中！
# ![](https://ai-studio-static-online.cdn.bcebos.com/4953c1e357e348669b919201c995c51c267a2f6db35b465080ebbddbd1b191b2)
# 
# 本教程是基于[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)实现的First Order Motion model, 实现让任何人唱起「蚂蚁呀嘿」的旋律,若是大家喜欢这个教程，请到[Github PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)主页点击star呀！下面就让我们一起动手实现吧！
# 
# 整体实现只有三步：
# 1. 下载PaddleGAN代码
# 2. 运行First Order Motion model的命令
# 3. 给视频加上声音
# 
# 看~~ 是不是相当简单！！接下来让我们一步步开始吧！
# 

# ## First Order Motion model原理
# First Order Motion model的任务是image animation，给定一张源图片，给定一个驱动视频，生成一段视频，其中主角是源图片，动作是驱动视频中的动作，源图像通常包含一个主体，驱动视频包含一系列动作。
# 
# 通俗来说，First Order Motion能够将给定的驱动视频中的人物A的动作迁移至给定的源图片中的人物B身上，生成全新的以人物B的脸演绎人物A的表情的视频。
# 
# 以人脸表情迁移为例，给定一个源人物，给定一个驱动视频，可以生成一个视频，其中主体是源人物，视频中源人物的表情是由驱动视频中的表情所确定的。通常情况下，我们需要对源人物进行人脸关键点标注、进行表情迁移的模型训练。
# 
# 但是这篇文章提出的方法只需要在同类别物体的数据集上进行训练即可，比如实现太极动作迁移就用太极视频数据集进行训练，想要达到表情迁移的效果就使用人脸视频数据集voxceleb进行训练。训练好后，我们使用对应的预训练模型就可以达到前言中实时image animation的操作。

# ## 下载PaddleGAN代码

# In[1]:


# 从github上克隆PaddleGAN代码
get_ipython().system('git clone https://gitee.com/paddlepaddle/PaddleGAN')


# In[7]:


# 安装所需安装包
get_ipython().run_line_magic('cd', 'PaddleGAN/')
get_ipython().system('pip install -r requirements.txt')
get_ipython().system('pip install imageio-ffmpeg')
get_ipython().run_line_magic('cd', 'applications/')


# In[8]:


get_ipython().system('mkdir output')


# ## 执行命令
# 大家可以上传自己准备的视频和图片，并在如下命令中的source_image参数和driving_video参数分别换成自己的图片和视频路径，然后运行如下命令，就可以完成动作表情迁移，程序运行成功后，会在ouput文件夹生成名为result.mp4的视频文件，该文件即为动作迁移后的视频。本项目中提供了原始图片和驱动视频供展示使用。具体的各参数使用说明如下
# - driving_video: 驱动视频，视频中人物的表情动作作为待迁移的对象
# - source_image: 原始图片，视频中人物的表情动作将迁移到该原始图片中的人物上
# - relative: 指示程序中使用视频和图片中人物关键点的相对坐标还是绝对坐标，建议使用相对坐标，若使用绝对坐标，会导致迁移后人物扭曲变形
# - adapt_scale: 根据关键点凸包自适应运动尺度
# 

# In[10]:


# !export PYTHONPATH=$PYTHONPATH:/home/aistudio/PaddleGAN && python -u tools/first-order-demo.py  --driving_video ~/work/fullbody.MP4  --source_image ~/work/秃头乔哥.png --relative --adapt_scale
get_ipython().system('export PYTHONPATH=$PYTHONPATH:/home/aistudio/PaddleGAN && python -u tools/first-order-demo.py  --driving_video ~/work/fullbody.MP4  --source_image ~/work/1.jpg --relative --adapt_scale')


# ## 使用moviepy为生成的视频加上音乐

# In[5]:


# add audio
get_ipython().system('pip install moviepy')


# In[5]:


#为生成的视频加上音乐
from moviepy.editor import *

videoclip_1 = VideoFileClip("/home/aistudio/work/fullbody.MP4")
videoclip_2 = VideoFileClip("./output/result.mp4")

audio_1 = videoclip_1.audio

videoclip_3 = videoclip_2.set_audio(audio_1)

videoclip_3.write_videofile("./output/qiaoge.mp4", audio_codec="aac")


# # 到这里，蚂蚁呀嘿的制作就完成啦！！
# 
# 至此，本项目带大家体验了使用[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)完成**人脸表情迁移**的技术运用，欢迎大家尝试本项目进行不同类型的视频修复~ 
# 
# 当然，[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)的应用也不会止步于此，[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)还能提供各类不同的能力，包括**唇形合成（对嘴型）、视频/照片修复（上色、超分、插帧）、人脸动漫化、照片动漫化**等等！！一个比一个更厉害！
# 
# 强烈鼓励大家玩起来，激发[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)的潜能！
# 
# 记得点个Star收藏噢~~
