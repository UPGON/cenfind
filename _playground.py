import numpy as np
import cv2
import random


class MyClass:
    age = 0
    name = 'Leo'

    def __init__(self, age):
        self.age = age

    def foo(self, x):
        return f'I am a method self={self}'

    @classmethod
    def foo_classmethod(cls, x):
        return f'I am a classmethod cls={cls}'

    @staticmethod
    def foo_staticmethod(x):
        return f'I am a staticmethod x={x}'


# class Image:
#     def __init__(self, data):
#         self.data = data
#
#     def __repr__(self):
#         return f"Image({self.data.shape})"
#
#     def multiply(self, n=2):
#         return Image(self.data * n)


class Die:

    def __init__(self, sides=6):
        self.sides = sides

    def __get__(self, instance, hello=None):
        return int(random.random() * self.sides) + 1


class Game:
    die6 = Die()
    die10 = Die(sides=10)
    die12 = Die(sides=12)


if __name__ == '__main__':
    # somedata = np.ones((512, 512), dtype=np.uint8)
    # myimage = Image(somedata)
    #
    # doubled = myimage.multiply()
    # ddoubled = doubled.multiply()
    #
    # mask = np.zeros((512, 1024), dtype=np.uint8)
    # pt1, pt2 = (0, 8), (128, 256)
    # cv2.rectangle(mask, pt1, pt2, 255, -1)
    # cv2.putText(mask, f"Rectangle {pt1=} {pt2=}", pt2, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, thickness=2)
    # cv2.imwrite('out/test.png', mask)

    print(0)
