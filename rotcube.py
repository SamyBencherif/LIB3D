
from scene import *
import lib3d

class MyScene(Scene):

  def setup(self):
    lib3d.setSize(self.size)
    lib3d.toggleCube('test:Ruler', [0,0,3])
    lib3d.addTouchControls()
    #lib3d.rotation[0] = .4;
    #lib3d.rotation[1] = .4;

    lib3d.uiElements.append(lib3d.LinearJoystick(500,100, horizontal=1, autoReset=0))
    lib3d.scale = 300.;

  def update(self):
    background(0,0,0)
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


if __name__ == '__main__':
  run(MyScene())
