"""
Lib3D sample world
"""

from scene import *
from scene_drawing import *
import random
import math
import lib3d

class MyScene (Scene):

	def setup(self):
		lib3d.setSize(self.size)
		lib3d.outlines = None
		lib3d.addTouchControls()
		lib3d.load("wbuilder.dat")

	def did_change_size(self):
		lib3d.setSize(self.size)
	
	def update(self):
		# image('test:Sailboat', 0, 0, self.size.w, self.size.h)
		
		lib3d.render()
		lib3d.renderUI()
	
	@lib3d.access
	def touch_began(self, touch):	
		pass
	
	@lib3d.access
	def touch_moved(self, touch):
		pass
	
	@lib3d.access
	def touch_ended(self, touch):
		pass

if __name__ == '__main__':
	run(MyScene())
