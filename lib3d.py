"""
Cross platform isometric 3d game library in python

possible names: lib3d Q3D
"""

"""
TODO list

* gameplay
    - moveable character
    - goal area (with trigger)
    - collisions
* game objects
* advanced lighting
* improve save and load
* improve texture selection
* improve delete cube
"""

from scene_drawing import *
from scene import *
Button = None # not interested in scene.Button

from numpy import matrix, array
import numpy as np
from math import cos, sin, pi, asin, tan, sqrt

x_norm = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1]
]

y_norm = [
    [0,0,0],
    [1,0,0],
    [0,0,1],
    [1,0,1]
]

z_norm = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [1,1,0]
]

screen_size = (0,0)

quads = []

outlines = ([0.29, 0.03, 0.41], 1)

pointTestQueue = []
scale = 100

view_matrix = matrix([
    [1.,0.,0.],
    [0.,1.,0.],
    [1.,1.,1.]
])

inv_view_matrix = np.linalg.pinv(view_matrix)

rotation = [-.5,.5,0.]
position = [0.,0.,0.]
offset = [0.,0.]

uiElements = []

def addQuad(tex, points, position):
	quads.append([tex, (matrix(points).T + matrix([position,]*4).T)])
	
def toggleCube(tex, position):

	for n in range(3):
		norm = (x_norm, y_norm, z_norm)[n]
		for i in range(2):
		
			# generate vertices for face of cube
			vertices = (matrix(norm).T +
			matrix([[
			position[0]-.5+i*(n==0),
			position[1]-.5+i*(n==1),
			position[2]-.5+i*(n==2)
			],]*4).T)
			
			# only added it to the renderer if one doesn't already exist
			existing = getQuad(vertices)
			if not existing:
				quads.append([tex, vertices])
			else:
				# if a quad already exists, remove it
				# this is currently done by removing its texture
				existing[0] = None
				
def centerContent():
	meanPos = [0,0,0]
	for q in quads:
		vertices = q[1]
		meanVert = (vertices[:,0] + vertices[:,1] + vertices[:,2] + vertices[:,3])/4.
		meanPos += meanVert/len(quads)
	global position
	position = meanPos.T.tolist()[0]
	
def setSize(size):
	global screen_size
	screen_size = (size.w, size.h)
	
def printmat(mat):
	print('[')
	for i in range(mat.shape[0]):
		print('\t', end='')
		for j in range(mat.shape[1]):
			print('%.2f' % mat[i,j], end=' ')
		print('')
	print(']')
	
def zdepth(quadP):
	proj = view_matrix*quadP
	return sum([proj[2,i] for i in range(4)])
	
def pointTest(point, callback):
	pointTestQueue.append((point, callback))
	
def invProjectPoint(point2d):
	assert len(point2d) == 3 # please include z-depth coordinate. It can be zero.
	return inv_view_matrix * ((matrix(point2d).T - matrix([screen_size[0]/2, screen_size[1]/2, 0]).T) / scale)
	
def projectPoint(point3d):
	return matrix([screen_size[0]/2, screen_size[1]/2, 0]).T + view_matrix * matrix(point3d).T * scale
	
test_shader = Shader('''
precision highp float;
varying vec2 v_tex_coord;

// These uniforms are set automatically:
uniform sampler2D u_texture;
uniform float u_time;
uniform vec2 u_sprite_size;

// these are set by lib3d
uniform vec3 norm;
uniform vec3 pos;

void main(void) {
    vec2 uv = v_tex_coord;
    vec4 col = texture2D(u_texture,uv);

    float b = dot(norm, vec3(.7,.8,.9));
    gl_FragColor = vec4(col.rgb*b, col.a);
}
''')

# find nearest point on line defined by A and B
# to point T. Answer given as parametric coordinate
# this function is n-dimensional, coordinates must be
# column matrices
def findNearest(A,B,T):
	return (np.linalg.pinv(B-A) * (T-A)).tolist()[0][0]
	
def render():

	quads.sort(key=lambda q: zdepth(q[1]))
	
	pmat = matrix([position,]*4).T
	testHit = None
	i = 0
	use_shader(test_shader)
	while i < len(quads):
		if i >= len(quads):
			# caused by delete
			break
			
		q = quads[i]
		points2d = []
		texture = q[0]
		
		# skip deleted quads
		if not texture:
			del quads[i]
			continue
			
		pointMatrix = q[1]-pmat
		
		assert pointMatrix.shape == (3,4)
		
		pointMatrix = view_matrix * pointMatrix * scale
		
		assert pointMatrix.shape == (3,4)
		
		for j in range(pointMatrix.shape[1]):
			points2d.append(screen_size[0]/2+pointMatrix[0,j]+offset[0])
			points2d.append(screen_size[1]/2+pointMatrix[1,j]+offset[1])
			
		global pointTestQueue
		if pointTestQueue:
			testPoint, callback = pointTestQueue[0]
			# run test on testPoint and points2d
			
			# precise test
			A = matrix([[points2d[0]],
			[points2d[1]]])
			B = matrix([[points2d[2]],
			[points2d[3]]])
			C = matrix([[points2d[4]],
			[points2d[5]]])
			D = matrix([[points2d[6]],
			[points2d[7]]])
			M = (A+D)/2
			T = matrix([[testPoint[0]],
			[testPoint[1]]])
			
			failed = False
			for (P1,P2) in [(A,B),(B,D),(C,D),(A,C)]:
				control = P1 + (P2-P1) * findNearest(P1,P2,T)

				# if path from midpoint to testpoint crosses an edge...
				if findNearest(M, control, T) > 1:
					failed = True
					break
				
			if not failed:
				testHit = (q, points2d, callback)
				
		test_shader.set_uniform('norm', getNorm(q))
		test_shader.set_uniform('pos', getCenter(q[1]))
		image_quad(texture,*points2d)
		
		if outlines:
		
			stroke(*outlines[0])
			stroke_weight(outlines[1])
			
			line(points2d[0], points2d[1], points2d[2], points2d[3])
			line(points2d[2], points2d[3], points2d[6], points2d[7])
			line(points2d[6], points2d[7], points2d[4], points2d[5])
			line(points2d[4], points2d[5], points2d[0], points2d[1])
			
			no_stroke()
			
		i += 1
		
	use_shader(None)
	pointTestQueue = pointTestQueue[1:]
	if testHit:
		q, points2d, callback = testHit
		callback(q, points2d)
		
		
def getNorm(quad3d):
	tex, points = quad3d
	
	# if all X coordinates are the same return X norm
	if len(set(array(points)[0])) == 1:
		return array([1,0,0])
		
	# if all Y coordinates are the same return Y norm
	if len(set(array(points)[1])) == 1:
		return array([0,1,0])
		
	# if all Z coordinates are the same return Z norm
	if len(set(array(points)[2])) == 1:
		return array([0,0,1])
		
def getCenter(points):
	return array((points[:,0] + points[:,1] + points[:,2] + points[:,3]) / 4).T[0]
	
# find a quad with matching points
def getQuad(searchPoints):
	for q in quads:
		tex, points = q
		if not (points - searchPoints).any():
			return q
			
def rot2d(x,y,angle):
	return x*cos(angle) - y*sin(angle), x*sin(angle) + y*cos(angle)
	
def setViewMatrix():
	alpha = (rotation[0])
	beta = (rotation[1])
	
	# isometric projection
	view_matrix[0,0] = cos(beta)
	view_matrix[0,1] = 0
	view_matrix[0,2] = -sin(beta)
	view_matrix[1,0] = sin(alpha)*sin(beta)
	view_matrix[1,1] = cos(alpha)
	view_matrix[1,2] = sin(alpha)*cos(beta)
	view_matrix[2,0] = cos(alpha)*sin(beta)
	view_matrix[2,1] = -sin(alpha)
	view_matrix[2,2] = cos(alpha)*cos(beta)
	
	inv_view_matrix = np.linalg.pinv(view_matrix)
setViewMatrix()

def orbitalCam(touch):

	# screen y-movement -> x axis rotation
	rotation[0] -= (touch.location.y - touch.prev_location.y) / 100
	
	# screen x -> rot y
	rotation[1] += (touch.location.x - touch.prev_location.x) / 100
	
	setViewMatrix()
	
def panCam(touch):

	position[0] -= (touch.location.x - touch.prev_location.x)
	position[1] -= (touch.location.y - touch.prev_location.y)
	
class UIElement:

	def __init__(self):
		self.activeTouchId = None
		
	def hitTest(self, p):
		return False
		
	def touch_began(self, touch):
		self.activeTouchId = touch.touch_id
		
	def touch_moved(self, touch):
		pass
		
	def touch_ended(self, touch):
		self.activeTouchId = None
		
	def getValue(self):
		return None
		
	def render(self):
		pass
		
class ToggleGroup(UIElement):
	def __init__(self, *toggles):
		super().__init__()
		self.toggles = toggles
		if len(toggles):
			toggles[0].pressed = True
			
	def hitTest(self, p):
		return any([t.hitTest(p) for t in self.toggles])
		
	def touch_began(self, touch):
		super().touch_began(touch)
		for t in self.toggles:
			t.pressed = False
			if t.hitTest(touch.location):
				t.touch_began(touch)
				
	def touch_ended(self, touch):
		super().touch_ended(touch)
		for t in self.toggles:
			if t.activeTouchId == touch.touch_id:
				t.touch_ended(touch)
				
	def getValue(self):
		for t in self.toggles:
			if t.pressed:
				return t
				
	def render(self):
		for t in self.toggles:
			t.render()
			
class Button(UIElement):

	def __init__(self, x, y, w=100, h=70, label="Tap This", image=None, toggle=False):
		super().__init__()
		self.x = x; self.y = y
		self.w = w; self.h = h
		self.label = label
		self.image = image
		
		self.selected = False
		self.pressed = False
		
		self.toggle = toggle
		
	def render(self):
	
		if self.pressed and self.toggle:
			stroke(1,1,0,.7)
			stroke_weight(3)
		else:
			stroke(0,0,0,.4)
			stroke_weight(1)
			
		fill(1,1,1,.5 + .3*self.selected)
		
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		
		rect(x-self.w/2, y-self.h/2, self.w, self.h)
		if self.image:
			tint(1,1,1)
			s = min(self.w, self.h) - 20
			if s > 0:
				image(self.image, x-s/2, y-s/2, s, s)
				
		tint(.3,.3,.3)
		text(self.label, x=x, y=y, alignment=5, font_size=12)
		
	def touch_began(self, touch):
		super().touch_began(touch)
		self.selected = True
		
	def touch_ended(self, touch):
		super().touch_ended(touch)
		self.selected = False
		self.pressed = not self.pressed
		
	def getValue(self):
		v = self.pressed
		if not self.toggle:
			self.pressed = False
		return v
		
	def hitTest(self, p):
	
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		
		return x-self.w/2 < p.x < x+self.w/2 and y-self.h/2 < p.y < y+self.h/2
		
class LinearJoystick(UIElement):

	def __init__(self, x, y, horizontal=False, startingValue=0, autoReset=True):
		super().__init__()
		
		self.x = x
		self.y = y
		
		self.axis = ['vertical', 'horizontal'][horizontal]
		self.value = startingValue
		
		self.startingValue = startingValue
		self.autoReset = autoReset
		
		if horizontal:
			self.bw = 20
			self.bh = 40
		else:
			self.bw = 40
			self.bh = 20
			
	def hitTest(self, p):
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		
		if self.axis == "vertical":
			return (x-self.bw/2 < p.x < x+self.bw/2 and
			y-50 < p.y < y+50)
		else:
			return (x-50 < p.x < x+50 and
			y-self.bh/2 < p.y < y+self.bh/2)
			
	def touch_moved(self, touch):
		if self.axis == "vertical":
			self.value += (touch.location.y - touch.prev_location.y)/50
		else:
			self.value += (touch.location.x - touch.prev_location.x)/50
		self.value = max(-1, min(self.value, 1))
		
	def touch_ended(self, touch):
		super().touch_ended(touch)
		
		if self.autoReset:
			self.value = self.startingValue
			
	def getValue(self):
		return self.value
		
	def render(self):
	
		stroke(0,0,0,.4)
		stroke_weight(1)
		
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		
		if self.axis == "vertical":
			# draw slider bar
			fill(1,1,1,.5)
			rect(x-5, y-50, 10, 100)
			
			# draw slider button
			if self.activeTouchId != None:
				fill(1,1,1,1)
			else:
				fill(1,1,1,.8)
			rect(x-self.bw/2, y-self.bh/2 + 50*self.value, self.bw, self.bh)
		else:
			# draw slider bar
			fill(1,1,1,.5)
			rect(x-50, y-5, 100, 10)
			
			# draw slider button
			if self.activeTouchId != None:
				fill(1,1,1,1)
			else:
				fill(1,1,1,.8)
			rect(x-self.bw/2 + 50*self.value, y-self.bh/2, self.bw, self.bh)
			
			
class Joystick(UIElement):
	def __init__(self, x, y):
		super().__init__()
		self.x = x; self.y = y
		self.px = 0; self.py = 0
		
	def render(self):
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		
		stroke(0,0,0,.4)
		stroke_weight(1)
		
		# base
		fill(1,1,1,.5)
		ellipse(x-50,self.y-50,100,100)
		# puck
		if self.activeTouchId != None:
			fill(1,1,1,1)
		else:
			fill(1,1,1,.8)
		ellipse(x-25+self.px,y-25+self.py,50,50)
		
	def getValue(self):
		return (self.px/30, self.py/30)
		
	def hitTest(self, p):
		x = self.x; y = self.y
		if x < 0: x += screen_size[0]
		if y < 0: y += screen_size[1]
		tx,ty=p.x,p.y
		return (tx-x)**2 + (ty-y)**2 < 2500
		
	def confinePuck(self):
		m = sqrt(self.px**2 + self.py**2)
		maxDist = 30
		
		if m > maxDist:
			self.px *= maxDist/m
			self.py *= maxDist/m
			
	def touch_moved(self, touch):
		self.px += touch.location.x - touch.prev_location.x
		self.py += touch.location.y - touch.prev_location.y
		self.confinePuck()
		
	def touch_ended(self, touch):
		super().touch_ended(touch)
		
		self.px = 0
		self.py = 0
		
""" Decorators """

def access(originalEvent):

	def injection(self, touch):
	
		cancelEv = False
		for e in uiElements:
		
			if e.hitTest(touch.location) and originalEvent.__name__ == "touch_began":
				e.touch_began(touch)
				cancelEv = True # block event
			if e.activeTouchId == touch.touch_id and originalEvent.__name__ == "touch_moved":
				e.touch_moved(touch)
				cancelEv = True # block event
			if e.activeTouchId == touch.touch_id and originalEvent.__name__ == "touch_ended":
				e.touch_ended(touch)
				cancelEv = True # block event
				
		if not (cancelEv and UIBlocks):
			originalEvent(self, touch)
			
	return injection
	
""" End Decorators """

# optionally, interactions with lib3d UI are invisible
# to the original program. it is as if UI is on a top
# layer seperate from the rest of the screen
UIBlocks = True

def renderUI():

	for e in uiElements:
		e.render()
		
	if rotJoy:
		rotation[1] += rotJoy.getValue()[0] / 30
		rotation[0] -= rotJoy.getValue()[1] / 30
		setViewMatrix()
		
	if zoomSlider:
		global scale
		scale += zoomSlider.getValue() * 5
		scale = min(max(10, scale), 500)
		
	if panJoy:
		offset[0] -= panJoy.getValue()[0] * 3
		offset[1] -= panJoy.getValue()[1] * 3
		
panJoy = None
zoomSlider = None
rotJoy = None

def addTouchControls():

	# pan
	global panJoy
	panJoy = Joystick(65,100)
	uiElements.append(panJoy)
	
	# zoom
	global zoomSlider
	zoomSlider = LinearJoystick(65,220)
	uiElements.append(zoomSlider)
	
	# rotate
	global rotJoy
	rotJoy = Joystick(-65,100)
	uiElements.append(rotJoy)
	
	
def save(path):
	open(path, 'wt').write(str(quads))
	
def load(path):
	global quads
	quads = eval(open(path, 'rt').read())

