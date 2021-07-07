
from scene import *
import lib3d

class MyScene(Scene):

  def setup(self):
    lib3d.setSize(self.size)
    lib3d.toggleCube('test:Ruler', [0,0,0])
    lib3d.addTouchControls()
    lib3d.uiElements.append(lib3d.Joystick(-65, 220))

  def update(self):
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