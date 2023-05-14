#!/usr/bin/env python

from abc import ABC, abstractproperty, abstractclassmethod
import os
import re
import copy
import tempfile
import numpy as np

class BaseRobot(ABC):
    def __init__(self, source_file, picture_path, env_id):
        self.picture_path = picture_path
        with open(source_file) as file:
            lines = file.readlines()
            self.source_text = "".join(lines)

        self.environment_id = env_id # defined in custom evn script
        self.body_parts = {}
        self.__collect_body_parts()

    @abstractproperty
    def description(self):
        return ""

    def __key_to_regex(self, key):
        regex = key.replace("$", "\$")
        regex = regex.replace("(", "\(")
        regex = regex.replace(")", "\)")

        return regex

    def create(self, body_part_mask, individual=([],[]), tmp_file=None):
        """
            Writes XML file (in place) - robot specific XML string with changed body part variables
        """

        if tmp_file == None:
            tmp_file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")

        _, body_part_adjustments = individual

        text = copy.deepcopy(self.source_text)
        
        # adjust body part variables
        idx = 0
        for i, key in enumerate(self.body_parts):
            regex = self.__key_to_regex(key)
            # set to desired body part length if True in mask ELSE set to a default value given by xml source file
            if body_part_mask[i]:
                text = re.sub(regex, str(body_part_adjustments.flatten()[idx]), text)
                idx += 1
            else:
                text = re.sub(regex, str(self.body_parts[key]), text)

        # compute body parts calculations
        calculations = re.findall(r'@.*@', text)
        for calc in calculations:
            _calc = calc.strip('@')
            text = re.sub(calc, str(eval(_calc)), text)

        # clear and change file in place
        tmp_file.seek(0)
        tmp_file.truncate()
        tmp_file.write(text)
        tmp_file.flush()

        return tmp_file

    def create_default(self):
        tmp_file = self.create(body_part_mask=np.zeros([len(self.body_parts)]))

        return tmp_file

    def __collect_body_parts(self):
        """
            Get names and default lengths of all changable body parts from XML
            source file
        """

        part_names = re.findall(r'\$[A-Za-z0-9_]+\([+-]?[0-9]*[.]?[0-9]+\)\$', self.source_text)
        for part in part_names:
            if not part in self.body_parts:
                self.body_parts[part] = float(part[part.find("(")+1 : part.find(")")]) # get between parenthesis

    @property
    def body_part_names(self):
        return list(self.body_parts.keys())

class StickAnt(BaseRobot):
    def __init__(self):
        DIR = os.path.dirname(__file__)
        source_file = DIR+"/custom_stick_ant.xml"
        picture_path = DIR+"/Basic-Ant"
        environment_id = "custom/CustomEnv-v0"

        super(StickAnt, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return \
"The simplest robot with body consisting of a single sphere and 4 one-part legs. Lengths of each leg can be adjusted by GA on its own.\n\
The simplest robot with body consisting of a single sphere and 4 one-part legs. Lengths of each leg can be adjusted by GA on its own.\n\
The simplest robot with body consisting of a single sphere and 4 one-part legs. Lengths of each leg can be adjusted by GA on its own.\n"

class AntV3(BaseRobot):
    def __init__(self):
        DIR = os.path.dirname(__file__)
        source_file = DIR+"/ant.xml"
        picture_path = DIR+"/Ant-v3"
        environment_id = "custom/CustomEnv-v0"

        super(AntV3, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return "More complex robot. Body made of a single sphere with 4 legs, each having 2 parts - 2 joints per leg (hip, knee)."

class SpotLike(BaseRobot):
    def __init__(self):
        DIR = os.path.dirname(__file__)
        source_file = DIR+"/spot_like.xml"
        picture_path = DIR+"/SpotLike"
        environment_id = "custom/CustomEnv-v0"

        super(SpotLike, self).__init__(source_file, picture_path, environment_id)

    @property
    def description(self):
        return \
"Homage to the greatest Spot from BostonDynamics.\n\
The robot has 4 legs made of 2 parts each (thigh and calf ended with fixed foot). \n\
Altogether there are 12 joints (12 actuators) - 2 for each hip free to rotate along X and Y axis and 1 for each knee along Y axis\n"
