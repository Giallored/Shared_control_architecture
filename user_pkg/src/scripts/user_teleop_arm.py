#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class SimpleKeyTeleop():
    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('usr_cmd/pose', Pose)
        self._pub_key = rospy.Publisher('key', String)
        self._hz = rospy.get_param('~hz', 10)

        self._forward_rate = rospy.get_param('~forward_rate', 0.1)
        self._backward_rate = rospy.get_param('~backward_rate', 0.1)
        self._rotation_rate = rospy.get_param('~rotation_rate', 0.02)
        self._last_pressed = {}
        self._orientation = 0
        self._position = [0,0,0]

    movement_bindings = {
            #x-axis
            ord('w'): ( 1, 0, 0, 0),  
            ord('s'): ( -1, 0, 0, 0),
            #y-axis
            ord('a'): ( 0, 1, 0, 0),
            ord('d'): ( 0, -1, 0, 0),
            #z-axis
            curses.KEY_UP:    (0, 0, 1,  0),
            curses.KEY_DOWN:  (0, 0,-1,  0),
            #rotations
            curses.KEY_LEFT:  ( 0, 0, 0,  1),
            curses.KEY_RIGHT: ( 0, 0, 0, -1),
        }

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self._set_pose()
            self._publish()
            rate.sleep()
    
    def _get_pose(self, position, orientation):
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.w = orientation
        return pose

    def _set_pose(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.4:
                keys.append(a)
        position = [0.0]*3
        orientation = 0.0
        for k in keys:
            movements = self.movement_bindings[k]
            position[0] += movements[0]
            position[1] += movements[1]
            position[2] += movements[2]
            orientation += movements[3]
        for i in range(3):
            if position[i] > 0:
                position[i] = position[i] * self._forward_rate
            else:
                position[i] = position[i] * self._backward_rate
        orientation = orientation * self._rotation_rate
        self._orientation = orientation
        self._position = position

    def _key_pressed(self, keycode):
        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Linear: (%f, %f, %f), Angular: %f' % 
                                   (self._position[0],self._position[1],self._position[2], self._orientation))
        self._interface.write_line(5, 'Use arrow keys to move, q to exit.')
        self._interface.refresh()

        pose = self._get_pose(self._position, self._orientation)
        self._pub_cmd.publish(pose)


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass