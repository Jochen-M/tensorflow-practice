# coding: utf8

import os
import random
import itertools

import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class GameObj:
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class GameEnv:
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        plt.imshow(a, interpolation='nearest')
        plt.pause(30)

    def reset(self):
        self.objects = []
        hero = GameObj(self.new_position(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        goal = GameObj(self.new_position(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        goal2 = GameObj(self.new_position(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        goal3 = GameObj(self.new_position(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        goal4 = GameObj(self.new_position(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal4)
        fire = GameObj(self.new_position(), 1, 1, 0, -1, 'fire')
        self.objects.append(fire)
        fire2 = GameObj(self.new_position(), 1, 1, 0, -1, 'fire')
        self.objects.append(fire2)
        state = self.render_env()
        self.state = state
        return state

    def move_hero(self, direction):
        hero = self.objects[0]
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        self.objects[0] = hero

    def new_position(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        current_positions = []
        for obj in self.objects:
            if (obj.x, obj.y) not in current_positions:
                current_positions.append((obj.x, obj.y))
        for pos in current_positions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def check_goal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameObj(self.new_position(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(GameObj(self.new_position(), 1, 1, 0, -1, 'fire'))

                return other.reward, False
        return 0.0, False

    def render_env(self):
        a = np.ones([self.sizeY+2, self.sizeX+2, 3])
        a[1:-1, 1:-1] = 0
        for item in self.objects:
            a[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1,
                item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        return a

    def step(self, action):
        self.move_hero(action)
        reward, done = self.check_goal()
        state = self.render_env()
        return state, reward, done


class Qnetwork:
    def __init__(self):
        self.scalar_input = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.image_input = tf.reshape(self.scalar_input, shape=[-1, 84, 84, 3])
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs=self.image_input, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs=self.conv3, num_outputs=512,
            kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None)


env = GameEnv(size=5)
