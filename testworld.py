
from scene import *
import lib3d

class MyScene(Scene):

  def setup(self):
    lib3d.setSize(self.size)
    lib3d.toggleCube('test:Peppers', [0,0,2])
    lib3d.toggleCube('test:Sailboat', [0,0,4])
    lib3d.toggleCube('test:Ruler', [0,0,6])
    lib3d.toggleCube('test:Pattern', [-2,0,2])
    lib3d.toggleCube('test:Lenna', [-2,0,4])
    lib3d.toggleCube('test:Bridge', [-2,0,6])


    lib3d.addTouchControls()

    lib3d.uiElements.append(lib3d.LinearJoystick(500,100, horizontal=1, autoReset=0, startingValue=1))
    lib3d.scale = 300.;

  def update(self):
    background(0,0,0)
    lib3d.render()
    lib3d.renderUI()

    tint(1,1,1)
    text(str(lib3d.position), x=800, y=100)

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


if __name__ == '__main__':
  run(MyScene())
