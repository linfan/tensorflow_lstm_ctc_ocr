# -*- coding: utf-8 -*-
 
import os
import StringIO
from PIL import Image, ImageFont, ImageDraw
#import pygame
import random
 
#pygame.init()

#font = pygame.font.Font(os.path.join("fonts", "times.ttf"), 36)
#font = ImageFont.truetype(os.path.join("fonts", "times.ttf"), 36)
for num in range(100,110):
	#vl = random.randint(0,255)
	vl = 0
	imblank = Image.new("RGB", (768, 64), (vl, vl, vl))
	#rtext = font.render(str(num), True, (0, 0, 0), (255, 255, 255))
	#pygame.image.save(rtext, "t.jpg")
	#im=Image.open("t.jpg") 
	fname = "bgs/{:08d}.jpg".format(num)
	#imblank.paste(im,(100,10)) 
	imblank.save(fname)
#os.remove('t.jpg')
