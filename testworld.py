
from scene import *
import lib3d

class MyScene(Scene):

  def setup(self):
    lib3d.setSize(self.size)

    # some floor
    for x in range(4):
      for y in range(4):
        if (x-y)%2 == 0:
          lib3d.toggleCube('textures/marble-white.jpg', [x,-2,y])
        else:
          lib3d.toggleCube('textures/marble-black.jpg', [x,-2,y])

    # some cubes
    lib3d.toggleCube('test:Peppers', [0,0,2])
    #lib3d.toggleCube('test:Sailboat', [0,0,4])
    #lib3d.toggleCube('test:Ruler', [0,0,6])
    #lib3d.toggleCube('test:Pattern', [-2,0,6])
    #lib3d.toggleCube('test:Lenna', [-2,0,4])
    #lib3d.toggleCube('test:Bridge', [-2,0,2])

    lib3d.position = [-1,0,0]

    lib3d.addTouchControls()

  def update(self):
    background(0,.5,.7)
    lib3d.render()
    lib3d.renderUI()

  def did_change_size(self):
    lib3d.setSize(self.size)

  @lib3d.access
  def touch_began(self, touch):
    pass

  @lib3d.access
  def touch_moved(self, touch):
    pass

  @lib3d.access
  def touch_ended(self, touch):
    pass

  @lib3d.access
  def key_down(self, key):
    pass

  @lib3d.access
  def key_up(self, key):
    pass

if __name__ == '__main__':
  run(MyScene())
