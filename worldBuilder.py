"""
Lib3D cross-platform world creator
"""

from scene import *
from scene_drawing import *
import random
import math
import lib3d

textures = [
	'plf:Ground_StoneCenter',
	'plf:Ground_GrassCenter',
	'plf:Tile_BoxCrate',
	'plf:Tile_BoxExplosive_used',
	'plf:Tile_BoxItem_disabled',
]

mainTexture = 'plf:Ground_StoneCenter'

class MyScene (Scene):

	def removeCube(self,q,p):

		if len(lib3d.quads) <= 6:
			return

		tex, coords = q

		quadA = (coords.T + lib3d.getNorm(q)).T
		quadB = (coords.T - lib3d.getNorm(q)).T

		qPos = lib3d.getCenter(coords)
		
		aPos = lib3d.getCenter(quadA)
		bPos = lib3d.getCenter(quadB)

		if float(lib3d.projectPoint(aPos)[2]) > float(lib3d.projectPoint(bPos)[2]):
			# quadB is furthest to camera and exists on the target cube
			# The center of the cube we want to delete is between quadB and qPos
			nPos = lib3d.getCenter(quadB)
		else:
			nPos = lib3d.getCenter(quadA)

		lib3d.toggleCube('plf:Ground_StoneCenter', tuple((qPos+nPos)/2))

	def addCube(self,q,p):
		tex, coords = q

		quadA = (coords.T + lib3d.getNorm(q)).T
		quadB = (coords.T - lib3d.getNorm(q)).T

		qPos = lib3d.getCenter(coords)
		
		aPos = lib3d.getCenter(quadA)
		bPos = lib3d.getCenter(quadB)

		if float(lib3d.projectPoint(aPos)[2]) > float(lib3d.projectPoint(bPos)[2]):
			# quadA is closet to camera and exists on the target cube
			# the best place to put a new cube is between quadA and qPos.
			nPos = lib3d.getCenter(quadA)
		else:
			nPos = lib3d.getCenter(quadB)

		lib3d.toggleCube(self.texSelect.getValue().image, tuple((qPos+nPos)/2))

	def setup(self):
		self.head = [0,0,0]
		lib3d.toggleCube(mainTexture, self.head)

		lib3d.setSize(self.size)
		lib3d.outlines = None
		
		lib3d.addTouchControls()

		# delete mode button
		self.deleteMode = lib3d.Button(-65, 220, label="delete mode", toggle=True)
		lib3d.uiElements.append(self.deleteMode)

		self.saveBtn = lib3d.Button(-170, -45, label="Save")
		self.loadBtn = lib3d.Button(-60, -45, label="Load")
		lib3d.uiElements.append(self.saveBtn)
		lib3d.uiElements.append(self.loadBtn)

		self.texSelect = lib3d.ToggleGroup(
			lib3d.Button(50, -50, 80, 80, label="Stone", image="plf:Ground_StoneCenter", toggle=True),
			lib3d.Button(50, -140, 80, 80, label="Dirt", image="plf:Ground_GrassCenter", toggle=True),
			lib3d.Button(50, -230, 80, 80, label="Box", image="plf:Tile_BoxCrate", toggle=True)
		)
		lib3d.uiElements.append(self.texSelect)

	def did_change_size(self):
		lib3d.setSize(self.size)
	
	def update(self):
		image('test:Sailboat', 0, 0, self.size.w, self.size.h)
		
		lib3d.render()
		lib3d.renderUI()

		tint(1,1,1)
		text("%d quads" % (len(lib3d.quads)), x=20, y=20, alignment=9)

		if self.saveBtn.getValue():
			lib3d.save("wbuilder.dat")
		if self.loadBtn.getValue():
			lib3d.load("wbuilder.dat")
	
	@lib3d.access
	def touch_began(self, touch):	
		if self.deleteMode.getValue():
			lib3d.pointTest((touch.location.x, touch.location.y), self.removeCube)
		else:
			lib3d.pointTest((touch.location.x, touch.location.y), self.addCube)
	
	@lib3d.access
	def touch_moved(self, touch):
		
		def highlight(q,p):
			stroke(.7,0,.7)
			stroke_weight(3)
			line(p[0], p[1], p[2], p[3])
			line(p[2], p[3], p[6], p[7])
			line(p[4], p[5], p[6], p[7])
			line(p[4], p[5], p[0], p[1])

		lib3d.pointTest((touch.location.x, touch.location.y), highlight)
	
	@lib3d.access
	def touch_ended(self, touch):
		pass

if __name__ == '__main__':
	run(MyScene())
